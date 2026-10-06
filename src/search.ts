import type { Where } from "chromadb";
import { openCollection } from "./chroma.ts";
import type { Config } from "./config.ts";
import { Embedder, MODALITIES, type Modality } from "./embedder.ts";
import { fmtTime } from "./windows.ts";

export interface SearchOptions {
  k: number;
  file?: string;
  modality?: Modality;
  json?: boolean;
}

export interface Hit {
  file: string;
  clipIndex: number;
  tStart: number;
  tEnd: number;
  modality: Modality;
  score: number;
}

export async function search(cfg: Config, query: string, opts: SearchOptions): Promise<Hit[]> {
  const collection = await openCollection(cfg);
  const total = await collection.count();
  if (total === 0) {
    console.log(`collection "${cfg.collection}" is empty - run: bun run db:ingest -- <folder>`);
    return [];
  }

  const embedder = new Embedder(cfg);
  const queryVector = await embedder.embedText(cfg.queryPrefix + query);
  const result = await collection.query({
    queryEmbeddings: [queryVector],
    nResults: Math.min(Math.max(opts.k * MODALITIES.length, opts.k + 3), total),
    where: buildWhere(opts),
  });

  const hits = dedupe(result.ids[0] ?? [], result.metadatas[0] ?? [], result.distances[0] ?? []);
  hits.sort((a, b) => b.score - a.score);
  const top = hits.slice(0, opts.k);

  if (opts.json) {
    console.log(JSON.stringify({ query, results: top }, null, 2));
    return top;
  }
  printHits(cfg, query, total, top);
  return top;
}

/** Overlapping windows and the three per-clip vectors collapse to one row per clip. */
function dedupe(ids: string[], metadatas: (Record<string, unknown> | null)[], distances: (number | null)[]): Hit[] {
  const best = new Map<string, Hit>();
  ids.forEach((id, i) => {
    const meta = metadatas[i];
    const distance = distances[i];
    if (!meta || distance === null || distance === undefined) return;
    const hit: Hit = {
      file: String(meta.file ?? id),
      clipIndex: Number(meta.clip_index ?? -1),
      tStart: Number(meta.t_start ?? 0),
      tEnd: Number(meta.t_end ?? 0),
      modality: String(meta.modality ?? "combined") as Modality,
      score: 1 - distance,
    };
    const key = `${hit.file}#${hit.clipIndex}`;
    const prior = best.get(key);
    if (!prior || hit.score > prior.score) best.set(key, hit);
  });
  return [...best.values()];
}

function buildWhere(opts: SearchOptions): Where | undefined {
  const clauses: Where[] = [];
  if (opts.modality) clauses.push({ modality: { $eq: opts.modality } });
  if (opts.file) {
    const rel = opts.file.replace(/\\/g, "/");
    clauses.push({
      $or: [{ file: { $eq: rel } }, { name: { $eq: rel.slice(rel.lastIndexOf("/") + 1) } }],
    });
  }
  if (clauses.length === 0) return undefined;
  return clauses.length === 1 ? clauses[0]! : { $and: clauses };
}

function printHits(cfg: Config, query: string, total: number, hits: Hit[]): void {
  console.log(`query: "${query}"   collection ${cfg.collection} (${total} vectors)`);
  if (hits.length === 0) {
    console.log("no matches for that filter");
    return;
  }
  const width = Math.max(...hits.map((h) => h.file.length));
  console.log("");
  hits.forEach((h, i) => {
    const range = `${fmtTime(h.tStart)}-${fmtTime(h.tEnd)}`;
    console.log(
      `${String(i + 1).padStart(2)}. ${h.file.padEnd(width)}  ${range.padEnd(15)}  ${h.modality.padEnd(8)}  ${h.score.toFixed(3)}`,
    );
  });
}
