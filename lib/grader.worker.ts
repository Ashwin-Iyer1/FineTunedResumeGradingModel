// Dedicated worker that owns the embedding model. Inference is CPU-heavy, so
// it stays off the main thread to keep the page responsive. Job vectors are
// precomputed (scripts/embed_jobs.mjs), so only the resume is embedded here.
//
// Protocol:
//   {type:"init", revision}   -> {type:"ready", threads} | {type:"error"}
//   {type:"embed", id, text}  -> {type:"embedded", id, vector, chunks, ms} | {type:"error", id}
//   one-time download          -> {type:"progress", file, pct}
import { env, pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { MODEL_ID, chunkText, htmlToText } from "./model";

env.allowLocalModels = false;

let extractor: FeatureExtractionPipeline | null = null;

// The model window; longer chunks are truncated by the tokenizer.
const MAX_TOKENS = 256;

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      if (self.crossOriginIsolated) {
        env.backends.onnx.wasm!.numThreads = Math.min(
          4,
          Math.max(1, Math.floor((navigator.hardwareConcurrency || 2) / 2))
        );
      }
      extractor = (await pipeline("feature-extraction", MODEL_ID, {
        dtype: "q8",
        // Pinned to the revision the job vectors were computed with.
        revision: msg.revision,
        // The Hub config can mark model_quantized.onnx as external-data, but
        // the uploaded file is self-contained with no `_data` sidecar.
        use_external_data_format: false,
        progress_callback: (p: { status: string; file?: string; loaded?: number; total?: number }) => {
          if (p.status === "progress" && p.total && p.total > 1_000_000) {
            self.postMessage({
              type: "progress",
              file: p.file,
              pct: Math.round(((p.loaded || 0) / p.total) * 100),
            });
          }
        },
      })) as FeatureExtractionPipeline;
      self.postMessage({ type: "ready", threads: env.backends.onnx.wasm?.numThreads ?? 1 });
    } else if (msg.type === "embed") {
      const t0 = performance.now();
      // Chunk-and-average: a resume normally overflows the 256-token window
      // and plain truncation would drop everything after the first section.
      const chunks = chunkText(htmlToText(msg.text));
      const res = await extractor!(chunks, { pooling: "mean", normalize: true });
      const dim = res.dims[res.dims.length - 1];
      const data = res.data as Float32Array;
      const vector = new Float32Array(dim);
      for (let c = 0; c < chunks.length; c++) {
        for (let i = 0; i < dim; i++) vector[i] += data[c * dim + i];
      }
      let norm = 0;
      for (let i = 0; i < dim; i++) norm += vector[i] * vector[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < dim; i++) vector[i] /= norm;
      const ms = Math.round(performance.now() - t0);

      const tokens = chunks.map((c) => extractor!.tokenizer.encode(c).length);
      self.postMessage(
        {
          type: "embedded",
          id: msg.id,
          vector,
          ms,
          chunks: chunks.map((text, i) => ({
            chars: text.length,
            tokens: tokens[i],
            truncated: tokens[i] > MAX_TOKENS,
            preview: text.slice(0, 90),
          })),
        },
        { transfer: [vector.buffer] }
      );
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      id: msg.type === "embed" ? msg.id : undefined,
      message: String((err as Error)?.message || err),
    });
  }
};
