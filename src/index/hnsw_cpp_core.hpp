#pragma once

#include <string>
#include <vector>

namespace vectordb {

struct SearchResult {
    int id;
    double distance;
};

struct LayerConnection {
    int node_id;
    int layer;
    std::vector<int> neighbors;
};

struct CsrLayer {
    int layer;
    std::vector<int> offsets;
    std::vector<int> neighbors;
};

struct CsrLayerView {
    int layer;
    const int* offsets;
    int offsets_len;
    const int* neighbors;
    int neighbors_len;
};

struct BuildStats {
    int vectors = 0;
    int dimensions = 0;
    int max_layer = 0;
    long long directed_edges = 0;
    double total_seconds = 0.0;
    double construction_seconds = 0.0;
    double search_seconds = 0.0;
    double greedy_search_seconds = 0.0;
    double candidate_search_seconds = 0.0;
    double prune_seconds = 0.0;
    double csr_export_seconds = 0.0;
    bool uses_squared_l2 = false;
    bool uses_float_l2_accumulation = false;
    int search_calls = 0;
    int greedy_search_calls = 0;
    int candidate_search_calls = 0;
    int visited_resizes = 0;
    bool uses_reusable_search_heaps = false;
    int search_heap_resizes = 0;
    bool uses_bounded_adjacency = false;
    bool uses_heuristic_neighbors = false;
    bool uses_heuristic_reverse_pruning = false;
    int adjacency_layers_allocated = 0;
    int max_observed_degree = 0;
};

struct BuildGraphResult {
    int entry_point;
    int max_layer;
    std::vector<int> levels;
    std::vector<LayerConnection> connections;
    std::vector<CsrLayer> layers;
    BuildStats build_stats;
};

std::vector<SearchResult> search_layer(
    const float* query,
    const float* vectors,
    int n_vectors,
    int dimension,
    const int* offsets,
    int offsets_len,
    const int* neighbors,
    int neighbors_len,
    const int* entry_points,
    int n_entry_points,
    int num_closest,
    const std::string& metric
);

std::vector<std::vector<SearchResult>> search_batch(
    const float* queries,
    int n_queries,
    const float* vectors,
    int n_vectors,
    int dimension,
    const CsrLayerView* layers,
    int n_layers,
    int entry_point,
    int max_layer,
    int k,
    int ef,
    const std::string& metric
);

std::vector<int> prune_connections(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const int* connection_ids,
    int n_connection_ids,
    int max_connections,
    const std::string& metric
);

std::vector<int> select_heuristic_neighbors(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const int* candidate_ids,
    int n_candidate_ids,
    int max_connections,
    const std::string& metric
);

BuildGraphResult build_graph(
    const float* vectors,
    int n_vectors,
    int dimension,
    const int* levels,
    int n_levels,
    int max_connections,
    int ef_construction,
    const std::string& metric,
    bool include_connections
);

}  // namespace vectordb
