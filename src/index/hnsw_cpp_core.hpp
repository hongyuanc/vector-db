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

struct BuildGraphResult {
    int entry_point;
    int max_layer;
    std::vector<int> levels;
    std::vector<LayerConnection> connections;
    std::vector<CsrLayer> layers;
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

BuildGraphResult build_graph(
    const float* vectors,
    int n_vectors,
    int dimension,
    const int* levels,
    int n_levels,
    int max_connections,
    int ef_construction,
    const std::string& metric
);

}  // namespace vectordb
