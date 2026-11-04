/// \file all_embedding_models.hpp
/// \brief get_auto_embedding_model func
/// \author FastFlowLM Team
/// \date 2025-10-23
/// \version 0.9.24
/// \note This is a header file for get_auto_embedding_model func
#pragma once
#include "AutoModel/model_frontend.hpp"
#include "model_list.hpp"

#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
#include "modeling_gemma_embedding.hpp"
#endif

inline std::pair<std::string, std::unique_ptr<AutoEmbeddingModel>> get_auto_embedding_model(const std::string& model_tag, flm_rt::device* npu_device_inst, model_list& models) {

    if (!models.is_model_supported(model_tag))
        throw std::invalid_argument("Unknown embedding model: " + model_tag);
    auto [tag, info] = models.get_model_info(model_tag);
    const std::string family = info.at("details").at("family");
    auto& registry = flm::frontend::FrontendRegistry::instance();
    if (info.value("model_kind", "generation") == "embedding")
        return {tag, registry.create_embedding(family, npu_device_inst)};
    if (family != "embed-gemma")
        throw std::invalid_argument("Model is not an embedding model: " + tag);
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    return {tag, std::make_unique<Gemma_Embedding>(npu_device_inst)};
#else
    throw std::runtime_error("The built-in embedding model is not supported in this build");
#endif
}
