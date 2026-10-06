import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import type { Collection } from "chromadb";
import { openCollection } from "./chroma.ts";
import { ensureDirs, type Config } from "./config.ts";
import { Embedder, MODALITIES, type Modality } from "./embedder.ts";
import { extractClip, hasAudioStream, probeDuration, removeClip } from "./ffmpeg.ts";
import { hash12, loadManifest, saveManifest } from "./manifest.ts";
import { fmtRange, planWindows } from "./windows.ts";

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

export async function ingest(cfg: Config, root: string, opts: IngestOptions): Promise<void> {
  await ensureDirs(cfg);
  const videos = findVideos(root);
  if (videos.length === 0) {
    console.log(`no video files (${VIDEO_EXT.join(", ")}) found under ${root}`);
    return;
  }

  const collection = await openCollection(cfg);
  const embedder = new Embedder(cfg);
  const manifest = await loadManifest(cfg.manifestPath);
  const pending: PendingRecord[] = [];
  let filesEmbedded = 0;
  let filesSkipped = 0;
  let recordsWritten = 0;

  for (const abs of videos) {
    const rel = relative(root, abs).replace(/\\/g, "/");
    const stat = statSync(abs);
    const prior = manifest[rel];
    if (!opts.fresh && prior && prior.size === stat.size && prior.mtimeMs === stat.mtimeMs) {
      filesSkipped++;
      continue;
    }

    const fileId = prior?.fileId ?? hash12(rel);
    console.log(`${rel}  (${(stat.size / 1e6).toFixed(1)} MB)`);
    const { clips, records } = await ingestFile(cfg, embedder, collection, abs, rel, fileId, pending, opts);
    await flush(collection, pending);

    manifest[rel] = {
      fileId,
      size: stat.size,
      mtimeMs: stat.mtimeMs,
      clips,
      embeddedAt: new Date().toISOString(),
    };
    await saveManifest(cfg.manifestPath, manifest);

    filesEmbedded++;
    recordsWritten += records;
    console.log(`  ${clips} clips -> ${records} vectors (collection now ${await collection.count()})`);
  }

  console.log(
    `\ndone: ${filesEmbedded} file(s) embedded, ${filesSkipped} unchanged, ${recordsWritten} vectors upserted`,
  );
}

async function ingestFile(
  cfg: Config,
  embedder: Embedder,
  collection: Collection,
  abs: string,
  rel: string,
  fileId: string,
  pending: PendingRecord[],
  opts: IngestOptions,
): Promise<{ clips: number; records: number }> {
  const duration = await probeDuration(abs);
  const withAudio = await hasAudioStream(abs);
  const windows = planWindows(duration, cfg.clipSeconds, cfg.overlapSeconds);
  const want: Modality[] = withAudio ? [...MODALITIES] : ["video"];
  let records = 0;

  for (let index = 0; index < windows.length; index++) {
    const window = windows[index]!;
    const media = await extractClip(cfg, abs, window, `${fileId}/c${String(index).padStart(4, "0")}`, withAudio);
    let vectors: Record<Modality, number[]>;
    try {
      vectors = await embedder.embedClip(media, want);
    } catch (err) {
      // a context-overflow error is only actionable if the clip is named
      throw new Error(`${rel} ${fmtRange(window)}: ${err instanceof Error ? err.message : err}`);
    }

    for (const modality of want) {
      pending.push({
        id: `${fileId}:${index}:${modality}`,
        embedding: vectors[modality],
        metadata: {
          file: rel,
          name: rel.slice(rel.lastIndexOf("/") + 1),
          file_id: fileId,
          clip_index: index,
          t_start: round2(window.start),
          t_end: round2(window.end),
          modality,
          fps: cfg.videoFps,
          duration: round2(duration),
        },
      });
      records++;
    }

    if (!opts.keepMedia) await removeClip(media.dir);
    console.log(`  ${fmtRange(window)}  ${want.join("+")}  ${media.frames.length} frames`);
    if (pending.length >= cfg.batchRecords) await flush(collection, pending);
  }

  if (!opts.keepMedia) await removeClip(join(cfg.mediaDir, fileId));

  return { clips: windows.length, records };
}

async function flush(collection: Collection, pending: PendingRecord[]): Promise<void> {
  if (pending.length === 0) return;
  // upsert, not add: re-ingesting a changed file reuses the same ids
  await collection.upsert({
    ids: pending.map((r) => r.id),
    embeddings: pending.map((r) => r.embedding),
    metadatas: pending.map((r) => r.metadata),
  });
  pending.length = 0;
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

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
