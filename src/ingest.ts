import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import type { Collection } from "chromadb";
import { openCollection } from "./chroma.ts";
import { ensureDirs, type Config } from "./config.ts";
import { Embedder, MODALITIES, type Modality } from "./embedder.ts";
import { extractClip, hasAudioStream, probeDuration, removeClip } from "./ffmpeg.ts";
import { hash12, loadManifest, saveManifest } from "./manifest.ts";
import { fmtRange, planWindows, type Window } from "./windows.ts";

const VIDEO_EXT = [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"];

export interface IngestOptions {
  fresh: boolean;
  keepMedia: boolean;
}

interface PendingRecord {
  id: string;
  embedding: number[];
  metadata: Record<string, string | number>;
}

interface FileJob {
  abs: string;
  rel: string;
  fileId: string;
  size: number;
  mtimeMs: number;
  duration: number;
  windows: Window[];
  want: Modality[];
  withAudio: boolean;
  /** windows of this file not yet finished; the file completes at 0 */
  left: number;
}

export async function ingest(cfg: Config, root: string, opts: IngestOptions): Promise<void> {
  await ensureDirs(cfg);
  const videos = findVideos(root);
  if (videos.length === 0) {
    console.log(`no video files (${VIDEO_EXT.join(", ")}) found under ${root}`);
    return;
  }

  const collection = await openCollection(cfg);
  const embedder = new Embedder(cfg);
  const model = await embedder.probeModel();
  const manifest = await loadManifest(cfg.manifestPath);
  const pending: PendingRecord[] = [];

  // Plan every file that needs embedding up front so the pool below can keep
  // the endpoint's slots busy across file boundaries.
  const jobs: FileJob[] = [];
  let filesSkipped = 0;
  for (const abs of videos) {
    const rel = relative(root, abs).replace(/\\/g, "/");
    const stat = statSync(abs);
    const prior = manifest[rel];
    if (!opts.fresh && prior && prior.size === stat.size && prior.mtimeMs === stat.mtimeMs) {
      filesSkipped++;
      continue;
    }
    const duration = await probeDuration(abs);
    const windows = planWindows(duration, cfg.clipSeconds, cfg.overlapSeconds);
    if (windows.length === 0) {
      console.log(`${rel}: ffprobe reports no duration, skipping`);
      filesSkipped++;
      continue;
    }
    const withAudio = await hasAudioStream(abs);
    jobs.push({
      abs,
      rel,
      fileId: prior?.fileId ?? hash12(rel),
      size: stat.size,
      mtimeMs: stat.mtimeMs,
      duration,
      windows,
      want: withAudio ? [...MODALITIES] : ["video"],
      withAudio,
      left: windows.length,
    });
    console.log(`${rel}  (${(stat.size / 1e6).toFixed(1)} MB)`);
  }

  if (jobs.length === 0) {
    console.log(`\ndone: 0 file(s) embedded, ${filesSkipped} unchanged, 0 vectors upserted`);
    return;
  }
  const totalWindows = jobs.reduce((n, j) => n + j.windows.length, 0);
  console.log(
    `embedding endpoint: ${model.id}${model.loaded ? "" : " (NOT LOADED)"}, ${model.slots} parallel slot${model.slots === 1 ? "" : "s"} - ${jobs.length} file(s), ${totalWindows} window(s)`,
  );

  let filesEmbedded = 0;
  let recordsWritten = 0;
  const work: { job: FileJob; index: number }[] = [];
  for (const job of jobs) for (let index = 0; index < job.windows.length; index++) work.push({ job, index });

  // One global queue over all windows, at most `slots` in flight: the endpoint
  // can only embed that many clips at once, so going wider just queues work
  // inside the server, and a single queue keeps the slots fed as files finish.
  await runPool(work, model.slots, async ({ job, index }) => {
    const window = job.windows[index]!;
    const media = await extractClip(cfg, job.abs, window, `${job.fileId}/c${String(index).padStart(4, "0")}`, job.withAudio);
    let vectors: Record<Modality, number[]>;
    try {
      vectors = await embedder.embedClip(media, job.want);
    } catch (err) {
      // a context-overflow error is only actionable if the clip is named
      throw new Error(`${job.rel} ${fmtRange(window)}: ${err instanceof Error ? err.message : err}`);
    }

    for (const modality of job.want) {
      pending.push({
        id: `${job.fileId}:${index}:${modality}`,
        embedding: vectors[modality],
        metadata: {
          file: job.rel,
          name: job.rel.slice(job.rel.lastIndexOf("/") + 1),
          file_id: job.fileId,
          clip_index: index,
          t_start: round2(window.start),
          t_end: round2(window.end),
          modality,
          fps: cfg.videoFps,
          duration: round2(job.duration),
        },
      });
    }
    if (!opts.keepMedia) await removeClip(media.dir);
    console.log(`  ${job.rel}  ${fmtRange(window)}  ${job.want.join("+")}  ${media.frames.length} frames`);

    if (--job.left === 0) {
      // flush before marking the file done, so the manifest never claims
      // vectors that are still only in `pending`
      await flush(collection, pending);
      if (!opts.keepMedia) await removeClip(join(cfg.mediaDir, job.fileId));
      manifest[job.rel] = {
        fileId: job.fileId,
        size: job.size,
        mtimeMs: job.mtimeMs,
        clips: job.windows.length,
        embeddedAt: new Date().toISOString(),
      };
      await saveManifest(cfg.manifestPath, manifest);
      const records = job.windows.length * job.want.length;
      filesEmbedded++;
      recordsWritten += records;
      console.log(`  ${job.rel}: ${job.windows.length} clips -> ${records} vectors (collection now ${await collection.count()})`);
    }
    if (pending.length >= cfg.batchRecords) await flush(collection, pending);
  });

  await flush(collection, pending);

  console.log(
    `\ndone: ${filesEmbedded} file(s) embedded, ${filesSkipped} unchanged, ${recordsWritten} vectors upserted`,
  );
}

async function flush(collection: Collection, pending: PendingRecord[]): Promise<void> {
  // splice (not `pending.length = 0` after the await): other windows keep
  // pushing records while an upsert is in flight
  const batch = pending.splice(0, pending.length);
  if (batch.length === 0) return;
  // upsert, not add: re-ingesting a changed file reuses the same ids
  await collection.upsert({
    ids: batch.map((r) => r.id),
    embeddings: batch.map((r) => r.embedding),
    metadatas: batch.map((r) => r.metadata),
  });
}

function findVideos(root: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const abs = join(root, entry.name);
    if (entry.isDirectory()) out.push(...findVideos(abs));
    else if (VIDEO_EXT.includes(entry.name.slice(entry.name.lastIndexOf(".")).toLowerCase())) out.push(abs);
  }
  return out.sort(naturalCompare);
}

/** Keeps 0001.mp4 ... 0010.mp4 in numeric rather than lexicographic order. */
function naturalCompare(a: string, b: string): number {
  const as = a.split(/(\d+)/);
  const bs = b.split(/(\d+)/);
  for (let i = 0; i < Math.max(as.length, bs.length); i++) {
    const x = as[i] ?? "";
    const y = bs[i] ?? "";
    if (x === y) continue;
    const nx = Number(x);
    const ny = Number(y);
    if (x !== "" && y !== "" && Number.isFinite(nx) && Number.isFinite(ny)) return nx - ny;
    return x < y ? -1 : 1;
  }
  return 0;
}

/** Runs `worker` over `items` with at most `limit` calls in flight. */
async function runPool<T>(items: T[], limit: number, worker: (item: T, index: number) => Promise<void>): Promise<void> {
  const lanes = Math.max(1, Math.min(limit, items.length));
  let next = 0;
  const lane = async (): Promise<void> => {
    for (;;) {
      const index = next++;
      if (index >= items.length) return;
      await worker(items[index]!, index);
    }
  };
  await Promise.all(Array.from({ length: lanes }, () => lane()));
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
