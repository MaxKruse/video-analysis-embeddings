import OpenAI from "openai";
import { mediaKeyFor, type Config } from "./config.ts";
import type { ClipMedia } from "./ffmpeg.ts";

export type Modality = "audio" | "video" | "combined";
export const MODALITIES: readonly Modality[] = ["audio", "video", "combined"];

type ImagePart = { type: "image_url"; image_url: { url: string } };
type AudioPart = { type: "input_audio"; input_audio: { url?: string; data?: string } };
type TextPart = { type: "text"; text: string };
type ContentPart = ImagePart | AudioPart | TextPart;
type ModelInfo = {
  id?: string;
  architecture?: { input_modalities?: string[] };
  /** Single-model llama.cpp builds report the slot count on the entry itself. */
  total_slots?: number;
  /** Multi-model preset builds carry the spawn command line and the ini text. */
  status?: { value?: string; args?: string[]; preset?: string };
};

/**
 * EmbeddingGemma 2 client over an OpenAI-compatible `/v1/embeddings` endpoint
 * (llama-server). Media is referenced by `file://` URL by default so request
 * bodies stay small; `MEDIA_TRANSPORT=base64` inlines the bytes instead.
 */
export class Embedder {
  private openai: OpenAI;

  constructor(private cfg: Config) {
    this.openai = new OpenAI({
      baseURL: cfg.embedBaseUrl,
      apiKey: cfg.embedApiKey,
      maxRetries: 2,
      timeout: 120_000,
    });
  }

  /**
   * llama-server reports the model, the modalities it accepts, whether it is
   * loaded (multi-model preset builds), and how many parallel slots serve it.
   */
  async probeModel(): Promise<{ id: string; modalities: string[]; loaded: boolean; slots: number }> {
    const base = this.cfg.embedBaseUrl.endsWith("/") ? this.cfg.embedBaseUrl : `${this.cfg.embedBaseUrl}/`;
    const url = new URL("models", base);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
    const body = (await res.json()) as { data?: ModelInfo[] };
    const chosen = (body.data ?? []).find((m) => m.id === this.cfg.embedModel) ?? body.data?.[0];
    if (!chosen?.id) throw new Error("endpoint reported no models");
    return {
      id: chosen.id,
      modalities: chosen.architecture?.input_modalities ?? [],
      // classic single-model builds have no status field and always serve the loaded model
      loaded: chosen.status ? chosen.status.value === "loaded" : true,
      slots: slotsFor(chosen),
    };
  }

  async embedText(text: string): Promise<number[]> {
    const vectors = await this.request([{ content: [{ type: "text", text }] }]);
    return vectorAt(vectors, 0);
  }

  /** One HTTP request per clip: each requested modality is its own input object. */
  async embedClip(media: ClipMedia, want: Modality[]): Promise<Record<Modality, number[]>> {
    const inputs = await Promise.all(want.map(async (m) => ({ content: await partsFor(this.cfg, media, m) })));
    const vectors = await this.request(inputs);
    const out = {} as Record<Modality, number[]>;
    want.forEach((modality, i) => (out[modality] = vectorAt(vectors, i)));
    return out;
  }

  private async request(inputs: { content: ContentPart[] }[]): Promise<number[][]> {
    const res = await this.openai.embeddings.create({
      model: this.cfg.embedModel,
      // llama-server's multimodal input shape extends the OpenAI embeddings
      // API, so it is not representable in the SDK's own input types
      input: inputs as unknown as string[],
      encoding_format: "float",
    });
    if (res.data.length !== inputs.length) {
      throw new Error(`embedding endpoint returned ${res.data.length} vectors for ${inputs.length} inputs`);
    }
    return res.data
      .sort((a, b) => a.index - b.index)
      .map((d) => {
        if (d.embedding.length !== this.cfg.dims) {
          throw new Error(`expected a ${this.cfg.dims}-d embedding, got ${d.embedding.length}-d (set EMBED_DIMS)`);
        }
        return d.embedding;
      });
  }
}

function vectorAt(vectors: number[][], index: number): number[] {
  const vector = vectors[index];
  if (!vector) throw new Error(`embedding endpoint omitted vector ${index}`);
  return vector;
}
/**
 * How many clips a llama-server can embed at once (its slots).
 * Detection order: `total_slots` (single-model builds), the `--parallel`/`-np`
 * value in `status.args` (the spawn command line), the `parallel = N` line in
 * `status.preset` (the ini text). Non-llama endpoints report nothing -> 1,
 * which keeps embedding sequential.
 */
function slotsFor(model: ModelInfo): number {
  if (Number.isInteger(model.total_slots) && (model.total_slots ?? 0) >= 1) return model.total_slots as number;
  const args = model.status?.args;
  if (args) {
    for (let i = 0; i < args.length; i++) {
      const a = args[i]!;
      if (a === "--parallel" || a === "-np") {
        const n = Number(args[i + 1]);
        if (Number.isInteger(n) && n >= 1) return n;
      } else if (a.startsWith("--parallel=")) {
        const n = Number(a.slice("--parallel=".length));
        if (Number.isInteger(n) && n >= 1) return n;
      }
    }
  }
  const line = model.status?.preset?.match(/^parallel\s*=\s*(\d+)\s*$/m);
  if (line?.[1]) return Number(line[1]);
  return 1;
}

async function partsFor(cfg: Config, media: ClipMedia, modality: Modality): Promise<ContentPart[]> {
  const frames = modality === "audio" ? [] : await Promise.all(media.frames.map((f) => imagePart(cfg, f)));
  const audio = modality === "video" || !media.audio ? [] : [await audioPart(cfg, media.audio)];
  return [...frames, ...audio];
}

async function imagePart(cfg: Config, path: string): Promise<ImagePart> {
  if (cfg.transport === "file") {
    return { type: "image_url", image_url: { url: `file://${mediaKeyFor(cfg, path)}` } };
  }
  const b64 = Buffer.from(await Bun.file(path).bytes()).toString("base64");
  return { type: "image_url", image_url: { url: `data:image/jpeg;base64,${b64}` } };
}

async function audioPart(cfg: Config, path: string): Promise<AudioPart> {
  if (cfg.transport === "file") {
    return { type: "input_audio", input_audio: { url: `file://${mediaKeyFor(cfg, path)}` } };
  }
  return { type: "input_audio", input_audio: { data: Buffer.from(await Bun.file(path).bytes()).toString("base64") } };
}
