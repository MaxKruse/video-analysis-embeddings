import OpenAI from "openai";
import { mediaKeyFor, type Config } from "./config.ts";
import type { ClipMedia } from "./ffmpeg.ts";

export type Modality = "audio" | "video" | "combined";
export const MODALITIES: readonly Modality[] = ["audio", "video", "combined"];

type ImagePart = { type: "image_url"; image_url: { url: string } };
type AudioPart = { type: "input_audio"; input_audio: { url?: string; data?: string } };
type TextPart = { type: "text"; text: string };
type ContentPart = ImagePart | AudioPart | TextPart;

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

  /** llama-server reports the loaded model and the modalities it accepts. */
  async probeModel(): Promise<{ id: string; modalities: string[] }> {
    const base = this.cfg.embedBaseUrl.endsWith("/") ? this.cfg.embedBaseUrl : `${this.cfg.embedBaseUrl}/`;
    const url = new URL("models", base);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
    const body = (await res.json()) as {
      data?: { id?: string; architecture?: { input_modalities?: string[] } }[];
    };
    const chosen = (body.data ?? []).find((m) => m.id === this.cfg.embedModel) ?? body.data?.[0];
    if (!chosen?.id) throw new Error("endpoint reported no models");
    return { id: chosen.id, modalities: chosen.architecture?.input_modalities ?? [] };
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
