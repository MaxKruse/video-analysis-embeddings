# video-analysis-embeddings

Semantic search over a folder of MP4s: ask in plain text, get **file + timestamp range**.

```
bun run db:ingest -- ./videos
bun run start -- --search "dog barking"
bun run db:clear
```

```
query: "dog barking"   collection video_clips (184 vectors)

 1. 0042.mp4  02:15-02:45  combined  0.631
 2. 0017.mp4  00:50-01:10  audio     0.588
```

## What it does

1. `db:ingest` walks a folder, cuts every video into **30 s windows with 5 s overlap**
   (ffmpeg), and asks an OpenAI-compatible endpoint for **three embeddings per clip**:
   `audio` (16 kHz mono WAV), `video` (1 fps JPEG frames) and `combined` (both).
2. Vectors go into a single Chroma collection (`video_clips`, cosine) with metadata
   `{file, name, file_id, clip_index, t_start, t_end, modality, fps, duration}`.
3. `start --search` embeds the query with the EmbeddingGemma 2 search prefix, queries
   Chroma with a candidate pool wider than k, collapses the three vectors of the same
   clip to one row, and keeps only the best clip of each video - so the top k are k
   distinct videos. `--file` drills into one video and returns its matching clips.

Three vectors per clip is deliberate: for sound-event queries ("dog barking")
the audio-only vector is not diluted by ~85 % video tokens, while `combined` still
carries the joint context.

### Parallel embedding

llama-server can serve several clips at once (`-np`/`parallel`). Ingest reads
`/v1/models` before it starts and auto-detects the slot count of the configured
model: `total_slots` on the entry (single-model builds), otherwise the `--parallel`
value in `status.args`, otherwise the `parallel = N` line in `status.preset`
(multi-model preset builds). It then keeps exactly that many clips in flight from a
single queue over the whole corpus - windows of later files start while earlier
files are still embedding, so the slots stay saturated and files with fewer windows
than slots never leave capacity idle. Endpoints without slot reporting default to 1
(sequential).

## Workflow - commands in order

```bash
# 0. once: config
cp .env.example .env

# 1. vector database
docker compose up -d

# 2. check: Chroma up, embedder preset present on the provider, slot count
bun run doctor

# 3. embed the corpus - re-runnable, only changed files are re-embedded
bun run db:ingest -- ./videos

# 4. query as often as you like, no re-embedding needed
bun run start -- --search "dog barking"
bun run start -- --search "dog barking" --modality audio --k 10
bun run start -- --search "dog barking" --file 0042.mp4 --json

# 5. reset: drops the collection, the manifest and the media cache
bun run db:clear
```

The embedding server is this machine's llama.cpp provider
(`http://127.0.0.1:8080/v1`) - a multi-model build serving presets from
`C:\Users\<you>\llama-cpp.ini`. It must be up with the `embeddinggemma-2` preset
loaded before ingest (start command in Setup); `doctor` fails loudly if the
preset is missing and warns if it is listed but not loaded.

`db:clear` also removes the manifest, so the next `db:ingest` re-embeds everything;
to force a re-embed without dropping search history use
`bun run db:ingest -- <folder> --fresh`.

## Requirements

- Bun 1.4+, ffmpeg/ffprobe on `PATH` (window + frame extraction)
- Docker (Chroma)
- A running **OpenAI-compatible embeddings server** with the multimodal
  EmbeddingGemma 2 model (`--embedding` mode, `--mmproj` projector)

## Setup

```bash
cp .env.example .env
docker compose up -d          # Chroma on http://127.0.0.1:8806
```

The embedding server is started from the ini, from the folder containing it:

```powershell
llama-server --models-preset .\llama-cpp.ini
```

The relevant preset, `[embeddinggemma-2]` in `C:\Users\<you>\llama-cpp.ini`:

```ini
[embeddinggemma-2]
model = C:\Users\<you>\LLMs\embeddinggemma-2\embeddinggemma-2-BF16.gguf
mmproj = C:\Users\<you>\LLMs\embeddinggemma-2\mmproj-BF16.gguf
embedding = true
ubatch-size = 8192
batch-size = 8192
no-webui = true
media-path = C:\Dev\video-analysis-embeddings\.cache\media
image-max-tokens = 140
parallel = 8
```

- `media-path` must point at `MEDIA_DIR` (default `.cache/media`) - that is how
  `MEDIA_TRANSPORT=file` resolves `file://` URLs (see Media transport).
- `parallel = 8` is what ingest picks up as the concurrency (see Parallel
  embedding). Change it in the preset, restart the provider, done.
- `image-max-tokens = 140` fixes the per-frame token cost the budget table below
  assumes.

Model files are in `C:\Users\<you>\LLMs\embeddinggemma-2\` -
`embeddinggemma-2-BF16.gguf` plus `mmproj-BF16.gguf` from
`unsloth/embeddinggemma-2-GGUF`.

Check both ends before ingesting:

```bash
bun run doctor     # prints resolved config, then pings Chroma and the endpoint
```

A healthy readout:

```
embeddings  ok (embeddinggemma-2, inputs: text+image+audio, 8 parallel slots)
```

## Commands

|Command|Flags|
|---|---|
|`bun run db:ingest -- <folder>`|`--fresh` (ignore the change manifest), `--keep-media` (keep extracted frames/WAV)|
|`bun run start -- --search "<q>"`|`--k N` (default 5, distinct videos), `--file NAME` (basename or relative path; returns the matching clips of that video), `--modality audio\|video\|combined`, `--json`|
|`bun run db:clear`|drops the collection, the manifest and the media cache|
|`bun run doctor`|-|

Ingest is incremental: a file is re-embedded only when its size or mtime changed
(`.cache/manifest.json`), and vectors are written with `upsert`, so re-running is
safe. A file's manifest entry is only saved after its vectors have been flushed to
Chroma, so an interrupted run never loses already-embedded files silently.
`.mp4 .m4v .mov .mkv .webm .avi` are picked up, recursively, in natural (numeric)
order. Videos without an audio track get the `video` vector only.

## Window size and the token budget

Each clip must fit into the **per-slot** context of the embedding model. At
1 fps a frame costs ~140 tokens (the preset's `image-max-tokens`) and audio
~25 tokens/s:

|Clip|Video tokens|Audio tokens|Total|
|---|---|---|---|
|30 s|4200|750|**4950**|
|45 s|6300|1125|7425|
|60 s|8400|1500|9900|

`CLIP_SECONDS=30` is the default. A short tail window is merged into the previous
one instead of being emitted as a runt, so a window can stretch to just under
40 s (~6600 tokens). If the server rejects a clip, ingest names it:

```
error: 0042.mp4 00:25-00:55: ... context size ...
```

then lower `CLIP_SECONDS` or give the preset a larger `ctx-size`.

## Media transport

`MEDIA_TRANSPORT=file` (default) sends `file://<path relative to MEDIA_DIR>` URLs,
e.g. `file://c8cb671a5f90/c0002/f0017.jpg`. That keeps request bodies tiny and
matches llama-server's `handle_media()`, which resolves such URLs against
`--media-path` (subdirectories allowed, `..` rejected).

`MEDIA_TRANSPORT=base64` inlines `data:image/jpeg;base64,...` instead. On the
llama.cpp builds tested both base64 forms failed against this model:

- data URI -> `400 {"message":"Invalid uri-encoded base64 value"}`
- raw base64 -> `500 Failed to load image or audio file`, with
  `mtmd_helper_video_init_from_buf: ffprobe failed on buffer (is ffprobe in PATH?)`

Use `file` unless you have a reason not to.

## Config

Everything is in `.env.example`. Notables: `EMBED_BASE_URL`, `EMBED_MODEL`
(must match the preset `id`/alias on the provider), `EMBED_DIMS` (768; ingest fails
loudly on a mismatch), `QUERY_PREFIX` (`task: search result | query: ` - required by
EmbeddingGemma 2 for queries, media gets no prefix), `CHROMA_URL` (`:8806`, next
free port in the `:88XX` scheme), `CLIP_SECONDS`, `OVERLAP_SECONDS`, `VIDEO_FPS`,
`AUDIO_RATE`. There is deliberately no concurrency setting: the slot count comes
from the endpoint itself.

## Status of verification

Verified end to end:

- against a deterministic OpenAI-compatible test double: ffmpeg windowing,
  frame/audio extraction, three vectors per clip, silent-video handling, Chroma
  writes, incremental re-ingest, `--file`/`--modality` filters, dedupe, timestamp
  output, `db:clear`, both media transports, and the slot-saturated pool (mock
  reports 4 slots -> 4 clips in flight, never more, across file boundaries);
- against the live multi-model provider at `:8080`: the `/v1/models` shape,
  auto-detected `parallel = 8` for the `embeddinggemma-2` preset, and the
  loaded/unloaded state in `doctor`.

**Not yet verified: embedding quality.** The multimodal payload shape, the 768-d
L2-normalized output and the `input_modalities` report were confirmed against live
llama-server builds; the semantic ranking itself has not been measured.

## Layout

```
src/cli.ts       arg parsing + dispatch (ingest | search | clear | doctor)
src/config.ts    .env -> Config
src/ffmpeg.ts    ffprobe probing, frame/WAV extraction
src/windows.ts   window planning, mm:ss formatting
src/embedder.ts  OpenAI SDK client, multimodal /v1/embeddings payload, slot detection
src/chroma.ts    collection bootstrap (cosine, no server-side embedding fn)
src/ingest.ts    per-corpus pipeline: plan, slot-saturated pool, batching, manifest
src/search.ts    query, dedupe, output
src/clear.ts     collection + manifest + media cache removal
```
