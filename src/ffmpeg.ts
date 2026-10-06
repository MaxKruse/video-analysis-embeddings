import { mkdir, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import type { Config } from "./config.ts";
import type { Window } from "./windows.ts";

export interface ClipMedia {
  dir: string;
  rel: string;
  frames: string[];
  audio: string | null;
}

async function run(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  const proc = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { code, stdout, stderr };
}

export async function probeDuration(video: string): Promise<number> {
  const r = await run([
    "ffprobe", "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    video,
  ]);
  if (r.code !== 0) throw new Error(`ffprobe failed for ${video}: ${r.stderr.trim()}`);
  const duration = Number(r.stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error(`ffprobe returned unusable duration for ${video}: "${r.stdout.trim()}"`);
  }
  return duration;
}

export async function hasAudioStream(video: string): Promise<boolean> {
  const r = await run([
    "ffprobe", "-v", "error",
    "-select_streams", "a:0",
    "-show_entries", "stream=index",
    "-of", "csv=p=0",
    video,
  ]);
  return r.code === 0 && r.stdout.trim().length > 0;
}

/**
 * Seeks with `-ss` before `-i` (fast + sample accurate when re-encoding) and
 * writes 1 fps JPEG frames plus 16 kHz mono PCM into the clip directory.
 */
export async function extractClip(
  cfg: Config,
  video: string,
  window: Window,
  rel: string,
  withAudio: boolean,
): Promise<ClipMedia> {
  const dir = join(cfg.mediaDir, rel);
  await mkdir(dir, { recursive: true });
  const length = window.end - window.start;

  const frames = await run([
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    "-ss", String(window.start), "-t", String(length),
    "-i", video,
    "-map", "0:v:0", "-an",
    "-vf", `fps=${cfg.videoFps}`,
    "-q:v", "2",
    join(dir, "f%03d.jpg"),
  ]);
  if (frames.code !== 0) {
    throw new Error(`ffmpeg frame extraction failed for ${video} @${window.start}: ${frames.stderr.trim()}`);
  }

  let audio: string | null = null;
  if (withAudio) {
    const a = await run([
      "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
      "-ss", String(window.start), "-t", String(length),
      "-i", video,
      "-vn", "-ac", "1", "-ar", String(cfg.audioRate),
      "-c:a", "pcm_s16le",
      join(dir, "audio.wav"),
    ]);
    if (a.code !== 0) {
      throw new Error(`ffmpeg audio extraction failed for ${video} @${window.start}: ${a.stderr.trim()}`);
    }
    audio = join(dir, "audio.wav");
  }

  const names = (await readdir(dir)).filter((n) => /^f\d{3}\.jpg$/.test(n)).sort();
  if (names.length === 0) throw new Error(`no frames extracted from ${video} @${window.start}`);

  return { dir, rel, frames: names.map((n) => join(dir, n)), audio };
}

export async function removeClip(dir: string): Promise<void> {
  await rm(dir, { recursive: true, force: true });
}
