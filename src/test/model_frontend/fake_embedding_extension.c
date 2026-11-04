/// \file fake_embedding_extension.c
/// \brief Independent C embedding provider for the dynamic-loader tests.
#include "AutoModel/model_extension.h"
#include <stdlib.h>
#include <string.h>

static int EndsWith(flm_span_v1 value, const char *suffix) {
    size_t n = strlen(suffix);
    return value.size >= n && !memcmp((const char *)value.data + value.size - n, suffix, n);
}
static int Load(void *object, flm_span_v1 path, flm_span_v1 info, uint32_t preemption,
                const flm_error_sink_v1 *error) {
    if (EndsWith(path, "missing")) abort(); /* A failed download must never reach load. */
    if (EndsWith(path, "fail-load")) return FLM_MODEL_EXTENSION_STATUS_ERROR;
    ((float *)object)[2] = EndsWith(path, "second") ? 2.0f : 0.5f;
    (void)info;
    (void)error;
    return path.size && !preemption ? 0 : FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
}
static int Embed(void *object, flm_span_v1 text, flm_span_v1 task, uint32_t dimensions,
                 flm_f32_view_v1 *output, const flm_error_sink_v1 *error) {
    float *values = (float *)object;
    (void)error;
    if (dimensions > 3)
        return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    values[0] = (float)text.size;
    values[1] = task.size == 8 && !memcmp(task.data, "document", 8) ? 2.0f : 1.0f;
    output->data = text.size == 0 ? NULL : values;
    output->size = dimensions ? dimensions : 3;
    if (EndsWith(text, "zero-output")) output->size = 0;
    if (EndsWith(text, "wrong-size")) output->size = 1;
    return 0;
}
static int Create(void *factory, const flm_frontend_create_context_v1 *context,
                  flm_embedding_model_v1 *model, const flm_error_sink_v1 *error) {
    (void)factory;
    (void)error;
    if (model->struct_size < FLM_EMBEDDING_MODEL_V1_MIN_SIZE)
        return 1;
    model->struct_size = sizeof(*model);
    model->object = calloc(3, sizeof(float));
    model->destroy = free;
    model->load = Load;
    model->embed = EndsWith(context->asset_root, "incomplete") ? NULL : Embed;
    return model->object ? 0 : 1;
}
static int Register(const flm_model_extension_registrar_v1 *registrar,
                    const flm_error_sink_v1 *error) {
    (void)error;
    if (registrar->struct_size < FLM_MODEL_EXTENSION_REGISTRAR_V1_EMBEDDING_SIZE)
        return FLM_MODEL_EXTENSION_STATUS_UNSUPPORTED;
    return registrar->register_embedding_provider(
        registrar->host_context, "test-embedding", "test-embedding", Create, NULL);
}
static const flm_model_extension_v1 extension = {sizeof(flm_model_extension_v1),
                                                 FLM_MODEL_EXTENSION_ABI_V1,
                                                 FLM_TEST_NPU_RUNTIME,
                                                 "test-embedding",
                                                 "1",
                                                 Register};
FLM_MODEL_EXTENSION_EXPORT const flm_model_extension_v1 *
flm_model_extension_query_v1(void) {
    return &extension;
}
