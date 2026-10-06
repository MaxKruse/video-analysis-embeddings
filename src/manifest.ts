import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";

export interface FileEntry {
  fileId: string;
  size: number;
  mtimeMs: number;
  clips: number;
  embeddedAt: string;
}

export type Manifest = Record<string, FileEntry>;

export async function loadManifest(path: string): Promise<Manifest> {
  const file = Bun.file(path);
  if (!(await file.exists())) return {};
  try {
    return (await file.json()) as Manifest;
  } catch {
    console.warn(`manifest ${path} is unreadable, starting fresh`);
    return {};
  }
}

export async function saveManifest(path: string, manifest: Manifest): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await Bun.write(path, JSON.stringify(manifest, null, 2));
}

export function hash12(text: string): string {
  return Bun.hash.wyhash(text).toString(16).padStart(16, "0").slice(0, 12);
}
