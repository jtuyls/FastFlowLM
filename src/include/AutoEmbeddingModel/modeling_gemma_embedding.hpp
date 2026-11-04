/// \file modeling_gemma_embedding.hpp
/// \brief Gemma_Embedding class
/// \author FastFlowLM Team
/// \date 2025-10-23
/// \version 0.9.24
/// \note This is a header file for the Gemma_Embedding class

#include "auto_embedding_model.hpp"
#include "models/gemma_embedding/flm/aie2p/gemma_embedding.hpp"

class Gemma_Embedding : public AutoEmbeddingModel{
private:
    std::string _get_task_prefix(const std::string& task) {
        if (task.empty() || task == "query" || task == "BitextMining" || task == "Retrieval")
            return "task: search result | query: ";
        if (task == "document") return "title: none | text: ";
        if (task == "Clustering") return "task: clustering | query: ";
        if (task == "Classification" || task == "MultilabelClassification")
            return "task: classification | query: ";
        if (task == "CodeRetrieval") return "task: code retrieval | query: ";
        if (task == "SentenceSimilarity") return "task: sentence similarity | query: ";
        if (task == "Summarization") return "task: summarization | query: ";
        throw std::invalid_argument("Unsupported Gemma embedding task: " + task);
    }
public:
    Gemma_Embedding(flm_rt::device* npu_device_inst);
    ~Gemma_Embedding();

    void load_model(std::string model_path, json model_info, bool enable_preemption = false) override;
    std::vector<float> embed(std::string& text, const std::string& task = "",
                             uint32_t dimensions = 0) override;
};