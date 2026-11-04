# Embedding extensions

Embedding plugins use the same manifest, loader and lifetime rules as generation
plugins. After checking `FLM_MODEL_EXTENSION_REGISTRAR_V1_EMBEDDING_SIZE`, register
an embedding factory with `register_embedding_provider(family, provider, ...)`.
Its `flm_embedding_model_v1` owns preprocessing and inference; `embed` accepts
UTF-8 text, a task name and a requested dimension (0 means the model default), and
returns a provider-owned float32 view. The host copies that view before another
call. Unsupported tasks or dimensions return an error.

A catalog entry selects `model_kind: "embedding"`, `details.family` and
`frontend_provider`. It needs the normal download metadata, but no causal backend,
context or prefill fields. Existing entries default to `model_kind: "generation"`.
Optional `file_records` carries the same per-file size/hash records as
`model_info.json`, allowing an extension to distribute new weights without editing
FLM's global metadata.

Load the manifest with `--model-extension`, then serve the embedding tag directly,
or name it in `/v1/embeddings`'s `model` field. String arrays are handled one input
at a time. `dimensions` is optional; the existing built-in embedding implementation
continues to use its default dimension. Calls and model switching are serialized.
Invalid selections and failed downloads leave the current model available. Once
loading a replacement begins, the old model is released; a failed load leaves
no model selected, so the next request retries. Errors use HTTP 400 for invalid
requests and HTTP 500 for download/provider failures.

`src/test/model_frontend/fake_embedding_extension.c` is a complete minimal C
provider. Its loader/ownership tests, embedding handler tests and catalog tests run in the existing
`model_extension_contracts` CI suite.
