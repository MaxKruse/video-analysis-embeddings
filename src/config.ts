import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";

export type MediaTransport = "file" | "base64";

export interface Config {
  chromaUrl: string;
  collection: string;
  dims: number;

  embedBaseUrl: string;
  embedApiKey: string;
  embedModel: string;
  queryPrefix: string;

  clipSeconds: number;
  overlapSeconds: number;
  videoFps: number;
  audioRate: number;

  mediaDir: string;
  manifestPath: string;
  transport: MediaTransport;

  batchRecords: number;
}

function num(name: string, fallback: number): number {
  const raw = Bun.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name} must be a number, got "${raw}"`);
  return value;
}

function path(name: string, fallback: string): string {
  const raw = Bun.env[name] || fallback;
  return isAbsolute(raw) ? raw : resolve(process.cwd(), raw);
}

export function loadConfig(): Config {
  const clipSeconds = num("CLIP_SECONDS", 30);
  const overlapSeconds = num("OVERLAP_SECONDS", 5);
  if (clipSeconds <= 0) throw new Error("CLIP_SECONDS must be > 0");
  if (overlapSeconds < 0 || overlapSeconds >= clipSeconds) {
    throw new Error("OVERLAP_SECONDS must be in [0, CLIP_SECONDS)");
  }

  const transport = (Bun.env.MEDIA_TRANSPORT || "file") as MediaTransport;
  if (transport !== "file" && transport !== "base64") {
    throw new Error(`MEDIA_TRANSPORT must be "file" or "base64", got "${transport}"`);
  }

  return {
    chromaUrl: Bun.env.CHROMA_URL || "http://127.0.0.1:8806",
    collection: Bun.env.COLLECTION || "video_clips",
    dims: num("EMBED_DIMS", 768),

    embedBaseUrl: Bun.env.EMBED_BASE_URL || "http://127.0.0.1:8080/v1",
    embedApiKey: Bun.env.EMBED_API_KEY || "none",
    embedModel: Bun.env.EMBED_MODEL || "embeddinggemma-2",
    queryPrefix: Bun.env.QUERY_PREFIX ?? "task: search result | query: ",

    clipSeconds,
    overlapSeconds,
    videoFps: num("VIDEO_FPS", 1),
    audioRate: num("AUDIO_RATE", 16000),

    mediaDir: path("MEDIA_DIR", ".cache/media"),
    manifestPath: path("MANIFEST_PATH", ".cache/manifest.json"),
    transport,

    batchRecords: num("BATCH_RECORDS", 96),
  };
}

export async function ensureDirs(cfg: Config): Promise<void> {
  await mkdir(cfg.mediaDir, { recursive: true });
  await mkdir(dirname(cfg.manifestPath), { recursive: true });
}

export function describeConfig(cfg: Config): string[] {
  return [
    `chroma      ${cfg.chromaUrl} (collection ${cfg.collection}, ${cfg.dims}d, cosine)`,
    `embeddings  ${cfg.embedBaseUrl} (model ${cfg.embedModel})`,
    `windows     ${cfg.clipSeconds}s clip, ${cfg.overlapSeconds}s overlap, ${cfg.videoFps} fps frames, ${cfg.audioRate} Hz mono audio`,
    `media       ${cfg.mediaDir} (transport ${cfg.transport}${cfg.transport === "file" ? `, llama-server needs --media-path ${cfg.mediaDir}` : ""})`,
    `manifest    ${cfg.manifestPath}${existsSync(cfg.manifestPath) ? "" : " (absent)"}`,
  ];
}

export function mediaKeyFor(cfg: Config, abs: string): string {
  const rel = abs.startsWith(cfg.mediaDir) ? abs.slice(cfg.mediaDir.length).replace(/^[\\/]+/, "") : abs;
  return rel.replace(/\\/g, "/");
}
