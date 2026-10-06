import { ChromaClient, type Collection } from "chromadb";
import type { Config } from "./config.ts";

export function chromaClient(cfg: Config): ChromaClient {
  const url = new URL(cfg.chromaUrl);
  const port = Number(url.port) || (url.protocol === "https:" ? 443 : 80);
  return new ChromaClient({ host: url.hostname, port, ssl: url.protocol === "https:" });
}

/** Embeddings are produced by llama-server, so Chroma must never embed itself. */
export async function openCollection(cfg: Config): Promise<Collection> {
  return chromaClient(cfg).getOrCreateCollection({
    name: cfg.collection,
    embeddingFunction: null,
    configuration: { hnsw: { space: "cosine" } },
    metadata: { app: "video-analysis-embeddings", dims: cfg.dims },
  });
}

export async function collectionNames(cfg: Config): Promise<string[]> {
  return (await chromaClient(cfg).listCollections()).map((c) => c.name);
}

export async function dropCollection(cfg: Config, name: string): Promise<void> {
  await chromaClient(cfg).deleteCollection({ name });
}
