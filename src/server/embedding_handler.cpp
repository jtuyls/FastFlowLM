/// \file embedding_handler.cpp
/// \brief Embedding requests and model switching, serialized for each server.
#include "rest_handler.hpp"

/// \brief Select an embedding model, preserving a working model until downloads succeed.
void RestHandler::ensure_embed_model_loaded(const std::string& model_tag) {
    const std::string tag = supported_models.rectify_model_tag(model_tag);
    if (!supported_models.is_model_supported(tag))
        throw std::invalid_argument("Unknown embedding model: " + model_tag);
    auto info = supported_models.get_model_info(tag).second;
    if (!is_embedding_model(info))
        throw std::invalid_argument("Model is not an embedding model: " + model_tag);
    if (auto_embedding_engine && current_embedding_tag == tag) return;
    switch (downloader.is_model_downloaded(tag, false, /*fast_check=*/true)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            if (!downloader.pull_model(tag, modelscope))
                throw std::runtime_error("Failed to download embedding model: " + tag);
            info = supported_models.get_model_info(tag).second;
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            throw std::invalid_argument("Embedding model is incompatible: " + tag);
    }
    // Release device resources before constructing the replacement. A failed
    // construction/load leaves no engine or tag, so the next request retries.
    auto_embedding_engine.reset();
    current_embedding_tag.clear();
    auto engine = get_auto_embedding_model(info, npu_device_inst);
    engine->load_model(supported_models.get_model_path(tag), info, preemption);
    auto_embedding_engine = std::move(engine);
    current_embedding_tag = tag;
}

///@brief Handle the embeddings request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_embeddings(const json& request,
                                   std::function<void(const json&)> send_response,
                                   StreamResponseCallback send_streaming_response) {
    try {
        std::lock_guard lock(embedding_mutex);
        std::string model = request.at("model");
        std::vector<std::string> inputs;

        if (request.at("input").is_string()) {
            inputs.push_back(request.at("input").get<std::string>());
        }
        else if (request.at("input").is_array()) {
            for (const auto& item : request.at("input")) {
                inputs.push_back(item.get<std::string>());
            }
        }
        else {
            throw std::invalid_argument("input must be a string or an array of strings");
        }

        if (inputs.empty())
            throw std::invalid_argument("input must be a string or a non-empty array of strings");

        const std::string task = request.value("task", std::string());
        uint32_t dimensions = 0;
        if (request.contains("dimensions")) {
            if (!request["dimensions"].is_number_integer() || request["dimensions"].get<int64_t>() <= 0 ||
                request["dimensions"].get<int64_t>() > UINT32_MAX)
                throw std::invalid_argument("dimensions must be a positive uint32");
            dimensions = request["dimensions"].get<uint32_t>();
        }
        ensure_embed_model_loaded(model);
        json embedding_data = json::array();
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::cout << "Embedding input[" << i << "]: " << "\n" << inputs[i] << std::endl;
            std::vector<float> embedding_result = this->auto_embedding_engine->embed(inputs[i], task, dimensions);
            embedding_data.push_back({
                {"object", "embedding"},
                {"embedding", embedding_result},
                {"index", i}
            });
        }

        json response = {
            {"object", "list"},
            {"data", embedding_data},
            {"model", model},
            {"usage", {
                {"prompt_tokens", 0},
                {"total_tokens", 0}
            }}
        };
        send_response(response);
    }
    catch (const std::exception& e) {
        const int code = dynamic_cast<const std::invalid_argument*>(&e) ||
                         dynamic_cast<const nlohmann::json::exception*>(&e) ? 400 : 500;
        send_response({{"error", {{"message", e.what()}, {"code", code}}}});
    }
}
