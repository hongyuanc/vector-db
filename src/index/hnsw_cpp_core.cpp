#include "hnsw_cpp_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <vector>

namespace vectordb {
namespace {

struct HeapItem {
    double distance;
    int id;
};

struct BuildNode {
    int level = 0;
    std::vector<std::vector<int>> connections;
};

struct MinHeapCompare {
    bool operator()(const HeapItem& left, const HeapItem& right) const {
        if (left.distance == right.distance) {
            return left.id > right.id;
        }
        return left.distance > right.distance;
    }
};

struct MaxHeapCompare {
    bool operator()(const HeapItem& left, const HeapItem& right) const {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    }
};

double euclidean_distance(const float* query, const float* vector, int dimension) {
    double squared_sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double diff = static_cast<double>(query[i]) - static_cast<double>(vector[i]);
        squared_sum += diff * diff;
    }
    return std::sqrt(squared_sum);
}

double cosine_similarity(const float* query, const float* vector, int dimension) {
    double dot = 0.0;
    double query_norm = 0.0;
    double vector_norm = 0.0;

    for (int i = 0; i < dimension; ++i) {
        const double q = static_cast<double>(query[i]);
        const double v = static_cast<double>(vector[i]);
        dot += q * v;
        query_norm += q * q;
        vector_norm += v * v;
    }

    if (query_norm == 0.0 || vector_norm == 0.0) {
        return 0.0;
    }

    return dot / (std::sqrt(query_norm) * std::sqrt(vector_norm));
}

double heap_distance(
    const float* query,
    const float* vectors,
    int vector_id,
    int dimension,
    bool use_euclidean
) {
    const float* vector = vectors + (static_cast<std::size_t>(vector_id) * dimension);
    if (use_euclidean) {
        return euclidean_distance(query, vector, dimension);
    }
    return -cosine_similarity(query, vector, dimension);
}

void ensure_layer(BuildNode& node, int layer) {
    if (layer < 0) {
        return;
    }
    const std::size_t required_size = static_cast<std::size_t>(layer + 1);
    if (node.connections.size() < required_size) {
        node.connections.resize(required_size);
    }
}

void add_unique_connection(std::vector<int>& connections, int neighbor_id) {
    if (std::find(connections.begin(), connections.end(), neighbor_id) == connections.end()) {
        connections.push_back(neighbor_id);
    }
}

std::vector<HeapItem> order_results(
    std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCompare>& results
) {
    std::vector<HeapItem> ordered;
    ordered.reserve(results.size());
    while (!results.empty()) {
        ordered.push_back(results.top());
        results.pop();
    }

    std::sort(ordered.begin(), ordered.end(), [](const HeapItem& left, const HeapItem& right) {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    });

    return ordered;
}

std::vector<int> select_closest_ids(const std::vector<HeapItem>& candidates, int max_connections) {
    const int output_size = std::min(max_connections, static_cast<int>(candidates.size()));
    std::vector<int> output;
    output.reserve(static_cast<std::size_t>(output_size));
    for (int i = 0; i < output_size; ++i) {
        output.push_back(candidates[i].id);
    }
    return output;
}

std::vector<int> prune_connection_vector(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const std::vector<int>& connection_ids,
    int max_connections,
    bool use_euclidean
) {
    if (node_id < 0 || node_id >= n_vectors || max_connections <= 0) {
        return {};
    }

    const float* node_vector = vectors + (static_cast<std::size_t>(node_id) * dimension);
    std::vector<HeapItem> candidates;
    candidates.reserve(connection_ids.size());

    for (int connection_id : connection_ids) {
        if (connection_id < 0 || connection_id >= n_vectors || connection_id == node_id) {
            continue;
        }

        const float* connection_vector =
            vectors + (static_cast<std::size_t>(connection_id) * dimension);
        const double distance = use_euclidean
            ? euclidean_distance(node_vector, connection_vector, dimension)
            : -cosine_similarity(node_vector, connection_vector, dimension);
        candidates.push_back({distance, connection_id});
    }

    std::sort(candidates.begin(), candidates.end(), [](const HeapItem& left, const HeapItem& right) {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    });

    return select_closest_ids(candidates, max_connections);
}

std::vector<HeapItem> search_mutable_layer(
    const float* query,
    const float* vectors,
    int n_vectors,
    int dimension,
    const std::vector<BuildNode>& nodes,
    const std::vector<int>& entry_points,
    int num_closest,
    int layer,
    bool use_euclidean
) {
    if (num_closest <= 0 || nodes.empty()) {
        return {};
    }

    std::vector<std::uint8_t> visited(nodes.size(), 0);
    std::priority_queue<HeapItem, std::vector<HeapItem>, MinHeapCompare> candidates;
    std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCompare> results;

    for (int entry_id : entry_points) {
        if (
            entry_id < 0 ||
            static_cast<std::size_t>(entry_id) >= nodes.size() ||
            visited[static_cast<std::size_t>(entry_id)]
        ) {
            continue;
        }

        visited[static_cast<std::size_t>(entry_id)] = 1;
        const double distance = heap_distance(query, vectors, entry_id, dimension, use_euclidean);
        candidates.push({distance, entry_id});
        results.push({distance, entry_id});
    }

    while (!candidates.empty()) {
        const HeapItem current = candidates.top();
        candidates.pop();

        if (static_cast<int>(results.size()) >= num_closest) {
            const double worst_distance = results.top().distance;
            if (current.distance > worst_distance) {
                break;
            }
        }

        if (
            current.id < 0 ||
            static_cast<std::size_t>(current.id) >= nodes.size() ||
            layer < 0 ||
            static_cast<std::size_t>(layer) >= nodes[static_cast<std::size_t>(current.id)].connections.size()
        ) {
            continue;
        }

        const std::vector<int>& neighbors =
            nodes[static_cast<std::size_t>(current.id)].connections[static_cast<std::size_t>(layer)];
        for (int neighbor_id : neighbors) {
            if (
                neighbor_id < 0 ||
                neighbor_id >= n_vectors ||
                static_cast<std::size_t>(neighbor_id) >= nodes.size() ||
                visited[static_cast<std::size_t>(neighbor_id)]
            ) {
                continue;
            }

            visited[static_cast<std::size_t>(neighbor_id)] = 1;
            const double distance = heap_distance(
                query,
                vectors,
                neighbor_id,
                dimension,
                use_euclidean
            );

            candidates.push({distance, neighbor_id});
            results.push({distance, neighbor_id});

            if (static_cast<int>(results.size()) > num_closest) {
                results.pop();
            }
        }
    }

    return order_results(results);
}

}  // namespace

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
) {
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("metric must be 'euclidean' or 'cosine'");
    }
    if (n_vectors <= 0 || dimension <= 0 || num_closest <= 0 || offsets_len <= 0) {
        return {};
    }

    const bool use_euclidean = metric == "euclidean";
    std::vector<std::uint8_t> visited(static_cast<std::size_t>(n_vectors), 0);

    std::priority_queue<HeapItem, std::vector<HeapItem>, MinHeapCompare> candidates;
    std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCompare> results;

    for (int i = 0; i < n_entry_points; ++i) {
        const int entry_id = entry_points[i];
        if (entry_id < 0 || entry_id >= n_vectors || visited[entry_id]) {
            continue;
        }

        visited[entry_id] = 1;
        const double distance = heap_distance(query, vectors, entry_id, dimension, use_euclidean);
        candidates.push({distance, entry_id});
        results.push({distance, entry_id});
    }

    while (!candidates.empty()) {
        const HeapItem current = candidates.top();
        candidates.pop();

        if (static_cast<int>(results.size()) >= num_closest) {
            const double worst_distance = results.top().distance;
            if (current.distance > worst_distance) {
                break;
            }
        }

        if (current.id < 0 || current.id + 1 >= offsets_len) {
            continue;
        }

        int begin = offsets[current.id];
        int end = offsets[current.id + 1];
        begin = std::max(0, std::min(begin, neighbors_len));
        end = std::max(begin, std::min(end, neighbors_len));

        for (int idx = begin; idx < end; ++idx) {
            const int neighbor_id = neighbors[idx];
            if (neighbor_id < 0 || neighbor_id >= n_vectors || visited[neighbor_id]) {
                continue;
            }

            visited[neighbor_id] = 1;
            const double distance = heap_distance(
                query,
                vectors,
                neighbor_id,
                dimension,
                use_euclidean
            );

            candidates.push({distance, neighbor_id});
            results.push({distance, neighbor_id});

            if (static_cast<int>(results.size()) > num_closest) {
                results.pop();
            }
        }
    }

    std::vector<HeapItem> ordered;
    ordered.reserve(results.size());
    while (!results.empty()) {
        ordered.push_back(results.top());
        results.pop();
    }

    std::sort(ordered.begin(), ordered.end(), [](const HeapItem& left, const HeapItem& right) {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    });

    std::vector<SearchResult> output;
    output.reserve(ordered.size());
    for (const HeapItem& item : ordered) {
        const double public_distance = use_euclidean ? item.distance : -item.distance;
        output.push_back({item.id, public_distance});
    }

    return output;
}

std::vector<int> prune_connections(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const int* connection_ids,
    int n_connection_ids,
    int max_connections,
    const std::string& metric
) {
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("metric must be 'euclidean' or 'cosine'");
    }
    if (
        vectors == nullptr ||
        n_vectors <= 0 ||
        dimension <= 0 ||
        node_id < 0 ||
        node_id >= n_vectors ||
        n_connection_ids <= 0 ||
        max_connections <= 0
    ) {
        return {};
    }

    const bool use_euclidean = metric == "euclidean";
    const float* node_vector = vectors + (static_cast<std::size_t>(node_id) * dimension);
    std::vector<HeapItem> candidates;
    candidates.reserve(static_cast<std::size_t>(n_connection_ids));

    for (int i = 0; i < n_connection_ids; ++i) {
        const int connection_id = connection_ids[i];
        if (connection_id < 0 || connection_id >= n_vectors || connection_id == node_id) {
            continue;
        }

        const float* connection_vector =
            vectors + (static_cast<std::size_t>(connection_id) * dimension);
        const double distance = use_euclidean
            ? euclidean_distance(node_vector, connection_vector, dimension)
            : -cosine_similarity(node_vector, connection_vector, dimension);
        candidates.push_back({distance, connection_id});
    }

    std::sort(candidates.begin(), candidates.end(), [](const HeapItem& left, const HeapItem& right) {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    });

    const int output_size = std::min(max_connections, static_cast<int>(candidates.size()));
    std::vector<int> output;
    output.reserve(static_cast<std::size_t>(output_size));
    for (int i = 0; i < output_size; ++i) {
        output.push_back(candidates[i].id);
    }

    return output;
}

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
) {
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("metric must be 'euclidean' or 'cosine'");
    }
    if (
        vectors == nullptr ||
        levels == nullptr ||
        n_vectors <= 0 ||
        dimension <= 0 ||
        n_levels != n_vectors ||
        max_connections <= 0 ||
        ef_construction <= 0
    ) {
        BuildGraphResult empty_result;
        empty_result.entry_point = -1;
        empty_result.max_layer = 0;
        return empty_result;
    }

    const bool use_euclidean = metric == "euclidean";
    BuildGraphResult result;
    result.entry_point = -1;
    result.max_layer = 0;
    result.levels.reserve(static_cast<std::size_t>(n_vectors));

    std::vector<BuildNode> nodes;
    nodes.reserve(static_cast<std::size_t>(n_vectors));

    int entry_point = -1;
    int current_max_layer = 0;

    for (int vector_id = 0; vector_id < n_vectors; ++vector_id) {
        const int node_level = levels[vector_id];
        if (node_level < 0) {
            throw std::invalid_argument("levels must be non-negative");
        }

        BuildNode node;
        node.level = node_level;
        node.connections.resize(static_cast<std::size_t>(node_level + 1));
        nodes.push_back(node);
        result.levels.push_back(node_level);

        if (entry_point == -1) {
            entry_point = vector_id;
            current_max_layer = node_level;
            continue;
        }

        const float* query = vectors + (static_cast<std::size_t>(vector_id) * dimension);
        std::vector<int> nearest = {entry_point};

        for (int layer = current_max_layer; layer > node_level; --layer) {
            const std::vector<HeapItem> candidates = search_mutable_layer(
                query,
                vectors,
                n_vectors,
                dimension,
                nodes,
                nearest,
                1,
                layer,
                use_euclidean
            );
            std::vector<int> next_nearest = select_closest_ids(candidates, 1);
            if (!next_nearest.empty()) {
                nearest = next_nearest;
            }
        }

        for (int layer = node_level; layer >= 0; --layer) {
            const std::vector<HeapItem> candidates = search_mutable_layer(
                query,
                vectors,
                n_vectors,
                dimension,
                nodes,
                nearest,
                ef_construction,
                layer,
                use_euclidean
            );

            const int layer_max_connections = layer == 0 ? max_connections * 2 : max_connections;
            std::vector<int> selected_neighbors =
                select_closest_ids(candidates, layer_max_connections);

            ensure_layer(nodes[static_cast<std::size_t>(vector_id)], layer);
            nodes[static_cast<std::size_t>(vector_id)]
                .connections[static_cast<std::size_t>(layer)] = selected_neighbors;

            for (int neighbor_id : selected_neighbors) {
                if (
                    neighbor_id < 0 ||
                    neighbor_id >= n_vectors ||
                    static_cast<std::size_t>(neighbor_id) >= nodes.size()
                ) {
                    continue;
                }

                BuildNode& neighbor_node = nodes[static_cast<std::size_t>(neighbor_id)];
                ensure_layer(neighbor_node, layer);
                std::vector<int>& neighbor_connections =
                    neighbor_node.connections[static_cast<std::size_t>(layer)];
                add_unique_connection(neighbor_connections, vector_id);

                if (static_cast<int>(neighbor_connections.size()) > layer_max_connections) {
                    neighbor_connections = prune_connection_vector(
                        vectors,
                        n_vectors,
                        dimension,
                        neighbor_id,
                        neighbor_connections,
                        layer_max_connections,
                        use_euclidean
                    );
                }
            }

            nearest = selected_neighbors;
        }

        if (node_level > current_max_layer) {
            current_max_layer = node_level;
            entry_point = vector_id;
        }
    }

    result.entry_point = entry_point;
    result.max_layer = current_max_layer;

    if (include_connections) {
        for (std::size_t node_id = 0; node_id < nodes.size(); ++node_id) {
            const BuildNode& node = nodes[node_id];
            for (std::size_t layer = 0; layer < node.connections.size(); ++layer) {
                if (node.connections[layer].empty()) {
                    continue;
                }

                LayerConnection connection;
                connection.node_id = static_cast<int>(node_id);
                connection.layer = static_cast<int>(layer);
                connection.neighbors = node.connections[layer];
                std::sort(connection.neighbors.begin(), connection.neighbors.end());
                result.connections.push_back(connection);
            }
        }
    }

    for (int layer = 0; layer <= current_max_layer; ++layer) {
        CsrLayer csr_layer;
        csr_layer.layer = layer;
        csr_layer.offsets.resize(static_cast<std::size_t>(n_vectors + 1), 0);

        for (int node_id = 0; node_id < n_vectors; ++node_id) {
            csr_layer.offsets[static_cast<std::size_t>(node_id)] =
                static_cast<int>(csr_layer.neighbors.size());

            const BuildNode& node = nodes[static_cast<std::size_t>(node_id)];
            if (static_cast<std::size_t>(layer) >= node.connections.size()) {
                continue;
            }

            std::vector<int> sorted_neighbors = node.connections[static_cast<std::size_t>(layer)];
            std::sort(sorted_neighbors.begin(), sorted_neighbors.end());
            csr_layer.neighbors.insert(
                csr_layer.neighbors.end(),
                sorted_neighbors.begin(),
                sorted_neighbors.end()
            );
        }

        csr_layer.offsets[static_cast<std::size_t>(n_vectors)] =
            static_cast<int>(csr_layer.neighbors.size());

        if (!csr_layer.neighbors.empty()) {
            result.layers.push_back(csr_layer);
        }
    }

    return result;
}

}  // namespace vectordb
