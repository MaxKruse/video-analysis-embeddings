import { rm } from "node:fs/promises";
import { chromaClient, collectionNames, dropCollection } from "./chroma.ts";
import type { Config } from "./config.ts";

export async function clear(cfg: Config): Promise<void> {
  const client = chromaClient(cfg);
  const existing = await collectionNames(cfg);
  if (existing.includes(cfg.collection)) {
    const count = await (await client.getCollection({ name: cfg.collection })).count();
    await dropCollection(cfg, cfg.collection);
    console.log(`dropped collection ${cfg.collection} (${count} vectors)`);
  } else {
    console.log(`collection ${cfg.collection} not present`);
  }

  await rm(cfg.manifestPath, { force: true });
  console.log(`removed manifest ${cfg.manifestPath}`);
  await rm(cfg.mediaDir, { recursive: true, force: true });
  console.log(`removed media cache ${cfg.mediaDir}`);
}
