/// \file model_extension_embedding.cpp
/// \brief Adapt complete embedding providers to AutoEmbeddingModel.
#include "AutoEmbeddingModel/auto_embedding_model.hpp"
#include "AutoModel/model_extension_frontend.hpp"
#include "model_extension_detail.hpp"

namespace flm::frontend {
namespace {
using namespace flm::extension::detail;

void Check(int status, const ErrorCapture &error) {
    if (status != FLM_MODEL_EXTENSION_STATUS_OK)
        throw std::runtime_error("Embedding provider: " + error.message);
}

class ExtensionEmbeddingModel final : public AutoEmbeddingModel {
  public:
    ExtensionEmbeddingModel(flm_rt::device *device, flm_embedding_model_v1 api,
                            std::shared_ptr<void> lifetime)
        : AutoEmbeddingModel(device), api_(api), lifetime_(std::move(lifetime)) {}
    ~ExtensionEmbeddingModel() override {
        destroy_provider_object(api_.destroy, api_.object);
    }
    void load_model(std::string path, json info, bool preemption) override {
        is_model_loaded = false;
        const std::string metadata = info.dump();
        ErrorCapture error;
        auto sink = error.sink();
        Check(api_.load(api_.object, span(path), span(metadata), preemption, &sink),
              error);
        model_path = std::move(path);
        is_model_loaded = true;
    }
    std::vector<float> embed(std::string &text, embedding_task_type_t task) override {
        return embed(text, task, 0);
    }
    std::vector<float> embed(std::string &text, embedding_task_type_t task,
                             uint32_t dimensions) override {
        if (!is_model_loaded)
            throw std::runtime_error("Embedding model is not loaded");
        static const char *tasks[] = {"query",
                                      "document",
                                      "BitextMining",
                                      "Clustering",
                                      "Classification",
                                      "CodeRetrieval",
                                      "MultilabelClassification",
                                      "SentenceSimilarity",
                                      "Retrieval",
                                      "Summarization"};
        if (static_cast<size_t>(task) >= std::size(tasks))
            throw std::invalid_argument("Unknown embedding task");
        flm_f32_view_v1 output{};
        ErrorCapture error;
        auto sink = error.sink();
        Check(api_.embed(api_.object, span(text), span(tasks[task]), dimensions,
                         &output, &sink),
              error);
        if (!output.data || !output.size || (dimensions && output.size != dimensions))
            throw std::runtime_error(
                "Embedding provider returned invalid dimensions or data");
        return {output.data, output.data + output.size};
    }

  private:
    flm_embedding_model_v1 api_;
    std::shared_ptr<void> lifetime_;
};
} // namespace

EmbeddingFactory
make_extension_embedding_factory(ExtensionFrontendRegistration registration,
                                 std::string root, std::string assets,
                                 std::shared_ptr<void> lifetime) {
    return [registration, root = std::move(root), assets = std::move(assets),
            lifetime = std::move(lifetime)](
               void *raw_device) -> std::unique_ptr<AutoEmbeddingModel> {
        auto *device = static_cast<flm_rt::device *>(raw_device);
        const flm_frontend_create_context_v1 context{
            sizeof(context), span(root), span(assets),
            native_runtime_device_handle(device), host_npu_runtime()};
        flm_embedding_model_v1 instance{};
        instance.struct_size = sizeof(instance);
        ErrorCapture error;
        auto sink = error.sink();
        try {
            Check(registration.create_embedding(registration.factory_context, &context,
                                                &instance, &sink),
                  error);
            if (instance.struct_size < FLM_EMBEDDING_MODEL_V1_MIN_SIZE ||
                !instance.object || !instance.destroy || !instance.load ||
                !instance.embed)
                throw std::runtime_error(
                    "Embedding factory returned an incomplete model");
            return std::make_unique<ExtensionEmbeddingModel>(device, instance,
                                                             lifetime);
        } catch (...) {
            destroy_provider_object(instance.destroy, instance.object);
            throw;
        }
    };
}
} // namespace flm::frontend
