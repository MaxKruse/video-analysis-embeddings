# video-analysis-embeddings

Semantic search over a folder of MP4s: ask in plain text, get **file + timestamp range**.

```
bun run db:ingest -- ./videos
bun run start -- --search "Ear blowing sounds"
bun run db:clear
```

```
query: "Ear blowing sounds"   collection video_clips (184 vectors)

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
   Chroma, collapses the three vectors of the same clip to one row, and prints the
   timestamp range.

Three vectors per clip is deliberate: for sound-event queries ("ear blowing sounds")
the audio-only vector is not diluted by ~85 % video tokens, while `combined` still
carries the joint context.

## Workflow - commands in order

```bash
# 0. once: config
cp .env.example .env

# 1. vector database
docker compose up -d

# 2. embedding server (you start it; this app never does)
llama-server -m models/embeddinggemma-2-Q8_0.gguf --mmproj models/mmproj-Q8_0.gguf \
  --embedding --host 127.0.0.1 --port 8081 -c 16384 -ub 2048 -ngl 99 \
  --media-path .cache/media --no-webui

# 3. optional but useful: pings Chroma and the embedding endpoint
bun run doctor

# 4. embed the corpus - re-runnable, only changed files are re-embedded
bun run db:ingest -- ./videos

# 5. query as often as you like, no re-embedding needed
bun run start -- --search "Ear blowing sounds"
bun run start -- --search "Ear blowing sounds" --modality audio --k 10
bun run start -- --search "dog barking" --file 0042.mp4 --json

# 6. reset: drops the collection, the manifest and the media cache
bun run db:clear
```

Steps 1 and 2 must be up before 4. Step 4 must have run before 5: an empty
collection makes 5 print a hint instead of results. Quote the folder if its name
contains glob characters, e.g. `bun run db:ingest -- "./a_lot_of_[0001-9999]"`.
`db:clear` also removes the manifest, so the next `db:ingest` re-embeds everything;
to force a re-embed without dropping search history use `bun run db:ingest -- <folder> --fresh`.

## Requirements

- Bun 1.4+, ffmpeg/ffprobe on `PATH` (window + frame extraction)
- Docker (Chroma)
- A running **OpenAI-compatible embeddings server** with a multimodal
  EmbeddingGemma 2 model. This app is only a client - it never starts a server.

## Setup

```bash
cp .env.example .env
docker compose up -d          # Chroma on http://127.0.0.1:8806
```

Step 2 of the workflow is the llama-server invocation this project expects:
llama.cpp >= b11456 (which added the `gemma-embedding2` arch), `--embedding` for
embedding mode, `--mmproj` for the vision + audio projector, and `--media-path`
pointing at `MEDIA_DIR` (default `.cache/media`) - see Media transport. `-ub 2048`
avoids the server capping the vision token budget:

```
W srv load_model: cap image_max_tokens (original=1120) to 256 (n_ubatch = 512)
W srv load_model: increase n_ubatch (-ub) to increase vision token budget
```

Check both ends before ingesting:

```bash
bun run doctor     # prints resolved config, then pings Chroma and the endpoint
```

## Commands

|Command|Flags|
|---|---|
|`bun run db:ingest -- <folder>`|`--fresh` (ignore the change manifest), `--keep-media` (keep extracted frames/WAV)|
|`bun run start -- --search "<q>"`|`--k N` (default 5), `--file NAME` (basename or relative path), `--modality audio\|video\|combined`, `--json`|
|`bun run db:clear`|drops the collection, the manifest and the media cache|
|`bun run doctor`|-|

Ingest is incremental: a file is re-embedded only when its size or mtime changed
(`.cache/manifest.json`), and vectors are written with `upsert`, so re-running is
safe. `.mp4 .m4v .mov .mkv .webm .avi` are picked up, recursively, in natural
(numeric) order. Videos without an audio track get the `video` vector only.

## Window size and the token budget

EmbeddingGemma 2 shares one 8192-token context across text, image and audio.
At 1 fps a frame costs ~140 tokens and audio ~25 tokens/s:

|Clip|Video tokens|Audio tokens|Total|
|---|---|---|---|
|30 s|4200|750|**4950**|
|45 s|6300|1125|7425|
|60 s|8400|1500|9900 (over budget)|

`CLIP_SECONDS=30` is therefore the default. A short tail window is merged into the
previous one instead of being emitted as a runt, so a window can stretch to just
under 40 s (<= ~6600 tokens) - still inside the budget. If the server rejects a
clip, ingest names it:

```
error: 0042.mp4 00:25-00:55: ... context size ...
```

then lower `CLIP_SECONDS` or raise `-c`.

## Media transport

`MEDIA_TRANSPORT=file` (default) sends `file://<path relative to MEDIA_DIR>` URLs,
e.g. `file://c8cb671a5f90/c0002/f0017.jpg`. That keeps request bodies tiny and
matches llama-server's `handle_media()`, which resolves such URLs against
`--media-path` (subdirectories allowed, `..` rejected).

`MEDIA_TRANSPORT=base64` inlines `data:image/jpeg;base64,...` instead. On llama.cpp
build 11456 both base64 forms failed against this model:

- data URI -> `400 {"message":"Invalid uri-encoded base64 value"}`
- raw base64 -> `500 Failed to load image or audio file`, with
  `mtmd_helper_video_init_from_buf: ffprobe failed on buffer (is ffprobe in PATH?)`

Use `file` unless you have a reason not to.

## Config

Everything is in `.env.example`. Notables: `EMBED_BASE_URL`, `EMBED_MODEL`,
`EMBED_DIMS` (768; ingest fails loudly on a mismatch), `QUERY_PREFIX`
(`task: search result | query: ` - required by EmbeddingGemma 2 for queries, media
gets no prefix), `CHROMA_URL` (`:8806`, next free port in the `:88XX` scheme),
`CLIP_SECONDS`, `OVERLAP_SECONDS`, `VIDEO_FPS`, `AUDIO_RATE`.

## Status of verification

Verified end to end against a deterministic OpenAI-compatible test double: ffmpeg
windowing, frame/audio extraction, three vectors per clip, silent-video handling,
Chroma writes, incremental re-ingest, `--file`/`--modality` filters, dedupe,
timestamp output, `db:clear`, both media transports.

**Not yet verified: embedding quality.** That needs your own `llama-server` (this
machine was not free to run it). The multimodal payload shape, the 768-d
L2-normalized output and the `input_modalities: [text, image, audio, video]`
report were confirmed against a live llama-server build 11456 before it was shut
down; the semantic ranking itself has not been measured.

## Layout

```
src/cli.ts       arg parsing + dispatch (ingest | search | clear | doctor)
src/config.ts    .env -> Config
src/ffmpeg.ts    ffprobe probing, frame/WAV extraction
src/windows.ts   window planning, mm:ss formatting
src/embedder.ts  OpenAI SDK client, multimodal /v1/embeddings payload
src/chroma.ts    collection bootstrap (cosine, no server-side embedding fn)
src/ingest.ts    per-file pipeline, batching, manifest
src/search.ts    query, dedupe, output
src/clear.ts     collection + manifest + media cache removal
```
