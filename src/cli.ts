import { resolve } from "node:path";
import { chromaClient } from "./chroma.ts";
import { describeConfig, loadConfig, type Config } from "./config.ts";
import { Embedder, MODALITIES, type Modality } from "./embedder.ts";
import { clear } from "./clear.ts";
import { ingest } from "./ingest.ts";
import { search, type SearchOptions } from "./search.ts";

const HELP = `video-analysis-embeddings

  bun run db:clear
  bun run db:ingest -- <folder-with-mp4s> [--fresh] [--keep-media]
  bun run start -- --search "<query>" [--k N] [--file NAME] [--modality audio|video|combined] [--json]
  bun run doctor

Environment (see .env.example): EMBED_BASE_URL, EMBED_MODEL, EMBED_DIMS, CHROMA_URL,
CLIP_SECONDS, OVERLAP_SECONDS, VIDEO_FPS, MEDIA_TRANSPORT, MEDIA_DIR.
`;

interface Flags {
  positional: string[];
  k: number;
  file?: string;
  modality?: Modality;
  json: boolean;
  fresh: boolean;
  keepMedia: boolean;
  search?: string;
}

function parseFlags(args: string[]): Flags {
  const flags: Flags = { positional: [], k: 5, json: false, fresh: false, keepMedia: false };
  const value = (arg: string, inline: string | undefined, next: string[] | undefined) =>
    inline ?? next?.shift() ?? "";

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    const eq = arg.indexOf("=");
    const inline = eq >= 0 ? arg.slice(eq + 1) : undefined;
    const name = eq >= 0 ? arg.slice(0, eq) : arg;

    switch (name) {
      case "--k":
        flags.k = Number(value(name, inline, args.slice(i + 1)));
        if (inline === undefined) i++;
        break;
      case "--file":
        flags.file = value(name, inline, args.slice(i + 1));
        if (inline === undefined) i++;
        break;
      case "--modality": {
        const modality = value(name, inline, args.slice(i + 1));
        if (inline === undefined) i++;
        if (!MODALITIES.includes(modality as Modality)) {
          throw new Error(`--modality must be one of ${MODALITIES.join(", ")}`);
        }
        flags.modality = modality as Modality;
        break;
      }
      case "--search":
        flags.search = value(name, inline, args.slice(i + 1));
        if (inline === undefined) i++;
        break;
      case "--json":
        flags.json = true;
        break;
      case "--fresh":
        flags.fresh = true;
        break;
      case "--keep-media":
        flags.keepMedia = true;
        break;
      default:
        if (name.startsWith("-")) throw new Error(`unknown flag ${name}`);
        flags.positional.push(arg);
    }
  }
  if (!Number.isInteger(flags.k) || flags.k <= 0) throw new Error("--k must be a positive integer");
  return flags;
}

async function doctor(cfg: Config): Promise<void> {
  console.log(describeConfig(cfg).join("\n"));

  try {
    const ms = await chromaClient(cfg).heartbeat();
    console.log(`chroma      ok (heartbeat ${ms})`);
  } catch (err) {
    console.log(`chroma      UNREACHABLE at ${cfg.chromaUrl} (${message(err)}) - docker compose up -d`);
  }

  try {
    const model = await new Embedder(cfg).probeModel();
    const inputs = model.modalities.length > 0 ? `, inputs: ${model.modalities.join("+")}` : "";
    console.log(`embeddings  ok (${model.id}${inputs})`);
    if (model.id !== cfg.embedModel) {
      console.log(`            note: endpoint serves "${model.id}", config asks for "${cfg.embedModel}"`);
    }
    if (model.modalities.length > 0 && model.modalities.every((m) => m === "text")) {
      console.log("            WARNING: text-only endpoint - image and audio clips will be rejected.");
      console.log("            If that is not your embedding server, another service owns this port.");
    }
  } catch (err) {
    console.log(`embeddings  UNREACHABLE at ${cfg.embedBaseUrl} (${message(err)})`);
  }
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function main(): Promise<void> {
  const [command, ...rest] = process.argv.slice(2);
  const cfg = loadConfig();
  const flags = parseFlags(rest);

  switch (command) {
    case "ingest": {
      const root = flags.positional[0] ?? ".";
      await ingest(cfg, resolve(process.cwd(), root), { fresh: flags.fresh, keepMedia: flags.keepMedia });
      return;
    }
    case "search": {
      const query = flags.search ?? flags.positional.join(" ");
      if (!query) {
        console.log('usage: bun run start -- --search "<query>"');
        process.exitCode = 2;
        return;
      }
      const opts: SearchOptions = { k: flags.k, file: flags.file, modality: flags.modality, json: flags.json };
      await search(cfg, query, opts);
      return;
    }
    case "clear":
      await clear(cfg);
      return;
    case "doctor":
      await doctor(cfg);
      return;
    default:
      console.log(HELP);
      process.exitCode = command ? 2 : 0;
  }
}

main().catch((err) => {
  console.error(`error: ${message(err)}`);
  process.exitCode = 1;
});
