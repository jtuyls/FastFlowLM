/// \file test_embedding_handler.cpp
/// \brief Exercise the production embedding request handler without a device or network.
#include "rest_handler.hpp"
#include "../phi4_core/test_support.hpp"
#include <fstream>
#include <future>
#include <map>

// Only the constructor and downloader are substituted. The real handler,
// catalog, factory and C plugin execute every request below.
RestHandler::RestHandler(model_list& models, ModelDownloader& downloads,
                         program_args_t& args, flm_rt::device* device)
    : npu_device_inst(device), supported_models(models), downloader(downloads),
      modelscope(args.modelscope), preemption(args.preemption) {}
RestHandler::~RestHandler() = default;
AutoDecisionModel::~AutoDecisionModel() = default;
ImageReader::~ImageReader() = default;
download_utils::CurlInitializer::CurlInitializer() = default;
download_utils::CurlInitializer::~CurlInitializer() = default;

namespace {
using Status = ModelDownloader::ModelStatus;
std::map<std::string, Status> statuses;
int pulls = 0;
bool pull_succeeds = false;
}
ModelDownloader::ModelDownloader(model_list& models) : supported_models(models) {}
ModelDownloader::ModelStatus ModelDownloader::is_model_downloaded(
    const std::string& tag, bool, bool fast_check) {
    TEST_REQUIRE(fast_check);
    return statuses.at(tag);
}
bool ModelDownloader::pull_model(const std::string&, bool, bool) {
    ++pulls;
    return pull_succeeds;
}

void test_embedding_handler() {
    auto& registry = flm::frontend::FrontendRegistry::instance();
    registry.load_extension(FLM_TEST_EMBEDDING_MANIFEST, {});
    nlohmann::json entries;
    for (const char* name : {"first", "second", "missing", "incompatible", "fail-load"}) {
        entries[name] = {{"1", {{"name", name}, {"model_kind", "embedding"},
            {"details", {{"family", "test-embedding"}}}, {"url", "https://example.invalid"},
            {"files", {"config.json"}}, {"flm_min_version", "1.0.0"}}}};
        statuses[std::string(name) + ":1"] = Status::Ready;
    }
    entries["generation"] = entries["first"];
    entries["generation"]["1"]["model_kind"] = "generation";
    entries["generation"]["1"]["details"]["family"] = "llama3";
    statuses["missing:1"] = Status::Missing;
    statuses["incompatible:1"] = Status::Incompatible;
    const auto directory = std::filesystem::path(FLM_TEST_EMBEDDING_MANIFEST).parent_path();
    std::string path = (directory / "handler-models.json").string(), root = directory.string();
    std::ofstream(path) << nlohmann::json({{"model_path", "models"}, {"models", entries}});
    model_list models(path, root);
    TEST_REQUIRE(models.is_model_supported("generation:1"));
    ModelDownloader downloads(models);
    program_args_t args{};
    RestHandler server(models, downloads, args, nullptr);
    const auto request = [&](const std::string& tag, json input = "hello") {
        json response;
        server.handle_embeddings({{"model", tag}, {"input", input}},
            [&](const json& value) { response = value; }, {});
        return response;
    };
    const auto check = [&](const std::string& tag, float marker) {
        const auto response = request(tag);
        TEST_REQUIRE(response.at("data").at(0).at("embedding") == json({5, 1, marker}));
    };
    check("first:1", 0.5f);
    check("Ollama/first", 0.5f);
    for (const char* tag : {"missing:1", "incompatible:1", "absent:1", "Ollama/absent", "generation:1"}) {
        const auto error = request(tag).at("error");
        TEST_REQUIRE(error.at("code") == (std::string(tag) == "missing:1" ? 500 : 400));
        check("first", 0.5f);
    }
    TEST_REQUIRE(pulls == 1);
    TEST_REQUIRE(request("fail-load:1").contains("error"));
    check("first", 0.5f); // Reload after releasing the old model for a failed load.
    statuses["second:1"] = Status::Outdated;
    TEST_REQUIRE(request("second").contains("error"));
    check("first", 0.5f);
    pull_succeeds = true;
    check("Ollama/second:1", 2.0f);
    TEST_REQUIRE(pulls == 3);
    statuses["second:1"] = Status::Ready;
    const auto batch = request("second", json::array({"hi", "hello"}));
    TEST_REQUIRE(batch.at("data").size() == 2);
    TEST_REQUIRE(batch.at("data").at(0).at("embedding").at(0) == 2);
    TEST_REQUIRE(request("second", 42).contains("error"));
    for (const auto& malformed : {json::object(), json{{"model", "second"}}}) {
        server.handle_embeddings(malformed, [](const json& response) {
            TEST_REQUIRE(response.at("error").at("code") == 400);
        }, {});
    }
    auto first = std::async(std::launch::async, [&] {
        for (int i = 0; i < 20; ++i) check("first", 0.5f);
    });
    auto second = std::async(std::launch::async, [&] {
        for (int i = 0; i < 20; ++i) check("second", 2.0f);
    });
    first.get();
    second.get();
    registry.unload_extensions();
}
