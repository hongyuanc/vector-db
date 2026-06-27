#include "hnsw_cpp_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace vectordb {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct HeapItem {
    double distance;
    int id;
};

struct BuildLayer {
    int max_connections = 0;
    std::vector<int> neighbors;
};

struct BuildNode {
    int level = 0;
    std::vector<BuildLayer> connections;
};

struct SearchScratch {
    std::vector<std::uint32_t> visited_generation;
    std::vector<HeapItem> candidate_heap;
    std::vector<HeapItem> result_heap;
    std::uint32_t generation = 1;
    int visited_resizes = 0;
    int search_heap_resizes = 0;

    void ensure_size(std::size_t size) {
        if (visited_generation.size() < size) {
            visited_generation.resize(size, 0);
            ++visited_resizes;
        }
    }

    void prepare_heaps(std::size_t candidate_capacity, std::size_t result_capacity) {
        candidate_heap.clear();
        result_heap.clear();

        if (candidate_heap.capacity() < candidate_capacity) {
            candidate_heap.reserve(candidate_capacity);
            ++search_heap_resizes;
        }
        if (result_heap.capacity() < result_capacity) {
            result_heap.reserve(result_capacity);
            ++search_heap_resizes;
        }
    }

    void next_generation() {
        if (generation == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(visited_generation.begin(), visited_generation.end(), 0);
            generation = 1;
            return;
        }
        ++generation;
    }

    bool is_visited(int id) const {
        return (
            id >= 0 &&
            static_cast<std::size_t>(id) < visited_generation.size() &&
            visited_generation[static_cast<std::size_t>(id)] == generation
        );
    }

    void mark_visited(int id) {
        visited_generation[static_cast<std::size_t>(id)] = generation;
    }
};

struct SearchInstrumentation {
    long long distance_evaluations = 0;
    long long visited_nodes = 0;
    int max_visited_nodes_per_search = 0;
    long long candidate_heap_pushes = 0;
    long long result_heap_pushes = 0;
};

struct DistanceInstrumentation {
    long long neighbor_selection_distance_evaluations = 0;
    long long prune_distance_evaluations = 0;
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

double squared_euclidean_distance(const float* query, const float* vector, int dimension) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    int i = 0;

    for (; i + 3 < dimension; i += 4) {
        const float diff0 = query[i] - vector[i];
        const float diff1 = query[i + 1] - vector[i + 1];
        const float diff2 = query[i + 2] - vector[i + 2];
        const float diff3 = query[i + 3] - vector[i + 3];
        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    float squared_sum = (sum0 + sum1) + (sum2 + sum3);
    for (; i < dimension; ++i) {
        const float diff = query[i] - vector[i];
        squared_sum += diff * diff;
    }
    return static_cast<double>(squared_sum);
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
        return squared_euclidean_distance(query, vector, dimension);
    }
    return -cosine_similarity(query, vector, dimension);
}

int layer_connection_limit(int layer, int max_connections) {
    return layer == 0 ? max_connections * 2 : max_connections;
}

void ensure_layer(
    BuildNode& node,
    int layer,
    int max_connections,
    int& adjacency_layers_allocated
) {
    if (layer < 0) {
        return;
    }
    const std::size_t required_size = static_cast<std::size_t>(layer + 1);
    if (node.connections.size() < required_size) {
        const std::size_t old_size = node.connections.size();
        node.connections.resize(required_size);
        for (std::size_t layer_index = old_size; layer_index < required_size; ++layer_index) {
            BuildLayer& build_layer = node.connections[layer_index];
            build_layer.max_connections = layer_connection_limit(
                static_cast<int>(layer_index),
                max_connections
            );
            build_layer.neighbors.reserve(static_cast<std::size_t>(build_layer.max_connections + 1));
            ++adjacency_layers_allocated;
        }
    }
}

void add_unique_connection(BuildLayer& layer, int neighbor_id) {
    std::vector<int>& connections = layer.neighbors;
    if (std::find(connections.begin(), connections.end(), neighbor_id) == connections.end()) {
        connections.push_back(neighbor_id);
    }
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

bool contains_id(const std::vector<int>& ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

std::vector<int> select_heuristic_neighbor_ids(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const std::vector<HeapItem>& candidates,
    int max_connections,
    bool use_euclidean,
    long long* distance_counter
) {
    if (
        vectors == nullptr ||
        node_id < 0 ||
        node_id >= n_vectors ||
        dimension <= 0 ||
        max_connections <= 0
    ) {
        return {};
    }

    const int output_size = std::min(max_connections, static_cast<int>(candidates.size()));
    std::vector<int> output;
    output.reserve(static_cast<std::size_t>(output_size));

    for (const HeapItem& candidate : candidates) {
        if (static_cast<int>(output.size()) >= max_connections) {
            break;
        }
        if (
            candidate.id < 0 ||
            candidate.id >= n_vectors ||
            candidate.id == node_id ||
            contains_id(output, candidate.id)
        ) {
            continue;
        }

        const float* candidate_vector =
            vectors + (static_cast<std::size_t>(candidate.id) * dimension);
        bool selected_is_closer = false;
        for (int selected_id : output) {
            const double selected_distance = heap_distance(
                candidate_vector,
                vectors,
                selected_id,
                dimension,
                use_euclidean
            );
            if (distance_counter != nullptr) {
                ++(*distance_counter);
            }
            if (selected_distance < candidate.distance) {
                selected_is_closer = true;
                break;
            }
        }

        if (!selected_is_closer) {
            output.push_back(candidate.id);
        }
    }

    for (const HeapItem& candidate : candidates) {
        if (static_cast<int>(output.size()) >= max_connections) {
            break;
        }
        if (
            candidate.id < 0 ||
            candidate.id >= n_vectors ||
            candidate.id == node_id ||
            contains_id(output, candidate.id)
        ) {
            continue;
        }
        output.push_back(candidate.id);
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
    bool use_euclidean,
    bool use_heuristic,
    long long* distance_counter
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
            ? squared_euclidean_distance(node_vector, connection_vector, dimension)
            : -cosine_similarity(node_vector, connection_vector, dimension);
        if (distance_counter != nullptr) {
            ++(*distance_counter);
        }
        candidates.push_back({distance, connection_id});
    }

    std::sort(candidates.begin(), candidates.end(), [](const HeapItem& left, const HeapItem& right) {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    });

    if (use_heuristic) {
        return select_heuristic_neighbor_ids(
            vectors,
            n_vectors,
            dimension,
            node_id,
            candidates,
            max_connections,
            use_euclidean,
            distance_counter
        );
    }

    return select_closest_ids(candidates, max_connections);
}

std::vector<HeapItem> order_heap_results(const std::vector<HeapItem>& results) {
    std::vector<HeapItem> ordered = results;
    std::sort(ordered.begin(), ordered.end(), [](const HeapItem& left, const HeapItem& right) {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    });
    return ordered;
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
    bool use_euclidean,
    SearchScratch& scratch,
    SearchInstrumentation* instrumentation
) {
    if (num_closest <= 0 || nodes.empty()) {
        return {};
    }

    scratch.ensure_size(nodes.size());
    scratch.next_generation();
    const std::size_t expected_results =
        static_cast<std::size_t>(std::max(1, num_closest) + 1);
    const std::size_t expected_candidates = std::max<std::size_t>(
        static_cast<std::size_t>(entry_points.size()),
        expected_results * 2
    );
    scratch.prepare_heaps(expected_candidates, expected_results);
    long long visited_this_search = 0;
    std::vector<HeapItem>& candidates = scratch.candidate_heap;
    std::vector<HeapItem>& results = scratch.result_heap;
    const MinHeapCompare min_heap_compare;
    const MaxHeapCompare max_heap_compare;

    for (int entry_id : entry_points) {
        if (
            entry_id < 0 ||
            static_cast<std::size_t>(entry_id) >= nodes.size() ||
            scratch.is_visited(entry_id)
        ) {
            continue;
        }

        scratch.mark_visited(entry_id);
        ++visited_this_search;
        if (instrumentation != nullptr) {
            ++instrumentation->visited_nodes;
            ++instrumentation->distance_evaluations;
        }
        const double distance = heap_distance(query, vectors, entry_id, dimension, use_euclidean);
        candidates.push_back({distance, entry_id});
        if (instrumentation != nullptr) {
            ++instrumentation->candidate_heap_pushes;
        }
        std::push_heap(candidates.begin(), candidates.end(), min_heap_compare);
        results.push_back({distance, entry_id});
        if (instrumentation != nullptr) {
            ++instrumentation->result_heap_pushes;
        }
        std::push_heap(results.begin(), results.end(), max_heap_compare);
    }

    while (!candidates.empty()) {
        std::pop_heap(candidates.begin(), candidates.end(), min_heap_compare);
        const HeapItem current = candidates.back();
        candidates.pop_back();

        if (static_cast<int>(results.size()) >= num_closest) {
            const double worst_distance = results.front().distance;
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
            nodes[static_cast<std::size_t>(current.id)].connections[static_cast<std::size_t>(layer)].neighbors;
        for (int neighbor_id : neighbors) {
            if (
                neighbor_id < 0 ||
                neighbor_id >= n_vectors ||
                static_cast<std::size_t>(neighbor_id) >= nodes.size() ||
                scratch.is_visited(neighbor_id)
            ) {
                continue;
            }

            scratch.mark_visited(neighbor_id);
            ++visited_this_search;
            if (instrumentation != nullptr) {
                ++instrumentation->visited_nodes;
                ++instrumentation->distance_evaluations;
            }
            const double distance = heap_distance(
                query,
                vectors,
                neighbor_id,
                dimension,
                use_euclidean
            );

            const bool result_can_improve =
                static_cast<int>(results.size()) < num_closest ||
                distance <= results.front().distance;
            if (!result_can_improve) {
                continue;
            }

            candidates.push_back({distance, neighbor_id});
            if (instrumentation != nullptr) {
                ++instrumentation->candidate_heap_pushes;
            }
            std::push_heap(candidates.begin(), candidates.end(), min_heap_compare);
            results.push_back({distance, neighbor_id});
            if (instrumentation != nullptr) {
                ++instrumentation->result_heap_pushes;
            }
            std::push_heap(results.begin(), results.end(), max_heap_compare);

            if (static_cast<int>(results.size()) > num_closest) {
                std::pop_heap(results.begin(), results.end(), max_heap_compare);
                results.pop_back();
            }
        }
    }

    if (instrumentation != nullptr) {
        instrumentation->max_visited_nodes_per_search = std::max(
            instrumentation->max_visited_nodes_per_search,
            static_cast<int>(visited_this_search)
        );
    }

    return order_heap_results(results);
}

std::vector<SearchResult> search_csr_layer(
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
    bool use_euclidean,
    SearchScratch& scratch
) {
    if (n_vectors <= 0 || dimension <= 0 || num_closest <= 0 || offsets_len <= 0) {
        return {};
    }

    scratch.ensure_size(static_cast<std::size_t>(n_vectors));
    scratch.next_generation();

    std::priority_queue<HeapItem, std::vector<HeapItem>, MinHeapCompare> candidates;
    std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCompare> results;

    for (int i = 0; i < n_entry_points; ++i) {
        const int entry_id = entry_points[i];
        if (entry_id < 0 || entry_id >= n_vectors || scratch.is_visited(entry_id)) {
            continue;
        }

        scratch.mark_visited(entry_id);
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
            if (neighbor_id < 0 || neighbor_id >= n_vectors || scratch.is_visited(neighbor_id)) {
                continue;
            }

            scratch.mark_visited(neighbor_id);
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
        const double public_distance = use_euclidean ? std::sqrt(item.distance) : -item.distance;
        output.push_back({item.id, public_distance});
    }

    return output;
}

bool public_result_less(
    const SearchResult& left,
    const SearchResult& right,
    bool use_euclidean
) {
    if (left.distance == right.distance) {
        return left.id < right.id;
    }
    if (use_euclidean) {
        return left.distance < right.distance;
    }
    return left.distance > right.distance;
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

    SearchScratch scratch;
    return search_csr_layer(
        query,
        vectors,
        n_vectors,
        dimension,
        offsets,
        offsets_len,
        neighbors,
        neighbors_len,
        entry_points,
        n_entry_points,
        num_closest,
        metric == "euclidean",
        scratch
    );
}

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
) {
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("metric must be 'euclidean' or 'cosine'");
    }

    std::vector<std::vector<SearchResult>> output;
    if (n_queries > 0) {
        output.reserve(static_cast<std::size_t>(n_queries));
    }

    if (
        queries == nullptr ||
        vectors == nullptr ||
        layers == nullptr ||
        n_queries <= 0 ||
        n_vectors <= 0 ||
        dimension <= 0 ||
        n_layers <= 0 ||
        entry_point < 0 ||
        entry_point >= n_vectors ||
        max_layer < 0 ||
        k <= 0
    ) {
        for (int query_index = 0; query_index < n_queries; ++query_index) {
            output.push_back({});
        }
        return output;
    }

    const int beam_width = std::max(ef, k);
    const bool use_euclidean = metric == "euclidean";
    SearchScratch search_scratch;
    search_scratch.ensure_size(static_cast<std::size_t>(n_vectors));

    std::vector<const CsrLayerView*> layer_by_number(static_cast<std::size_t>(max_layer + 1), nullptr);
    for (int layer_index = 0; layer_index < n_layers; ++layer_index) {
        const CsrLayerView& layer = layers[layer_index];
        if (layer.layer >= 0 && layer.layer <= max_layer) {
            layer_by_number[static_cast<std::size_t>(layer.layer)] = &layer;
        }
    }

    for (int query_index = 0; query_index < n_queries; ++query_index) {
        const float* query = queries + (static_cast<std::size_t>(query_index) * dimension);
        std::vector<int> nearest;
        nearest.push_back(entry_point);

        for (int layer = max_layer; layer > 0; --layer) {
            const CsrLayerView* layer_view = layer_by_number[static_cast<std::size_t>(layer)];
            if (layer_view == nullptr) {
                continue;
            }

            const std::vector<SearchResult> candidates = search_csr_layer(
                query,
                vectors,
                n_vectors,
                dimension,
                layer_view->offsets,
                layer_view->offsets_len,
                layer_view->neighbors,
                layer_view->neighbors_len,
                nearest.data(),
                static_cast<int>(nearest.size()),
                1,
                use_euclidean,
                search_scratch
            );
            if (!candidates.empty()) {
                nearest.clear();
                nearest.push_back(candidates[0].id);
            }
        }

        const CsrLayerView* base_layer = layer_by_number[0];
        if (base_layer == nullptr) {
            output.push_back({});
            continue;
        }

        std::vector<SearchResult> candidates = search_csr_layer(
            query,
            vectors,
            n_vectors,
            dimension,
            base_layer->offsets,
            base_layer->offsets_len,
            base_layer->neighbors,
            base_layer->neighbors_len,
            nearest.data(),
            static_cast<int>(nearest.size()),
            beam_width,
            use_euclidean,
            search_scratch
        );
        if (static_cast<int>(candidates.size()) > k) {
            candidates.resize(static_cast<std::size_t>(k));
        }
        output.push_back(candidates);
    }

    return output;
}

std::vector<std::vector<SearchResult>> search_segmented_batch(
    const float* queries,
    int n_queries,
    int dimension,
    const HnswSegmentView* segments,
    int n_segments,
    int k,
    int ef,
    int segment_search_k,
    const std::string& metric
) {
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("metric must be 'euclidean' or 'cosine'");
    }

    std::vector<std::vector<SearchResult>> output;
    if (n_queries > 0) {
        output.reserve(static_cast<std::size_t>(n_queries));
    }

    if (
        queries == nullptr ||
        segments == nullptr ||
        n_queries <= 0 ||
        dimension <= 0 ||
        n_segments <= 0 ||
        k <= 0
    ) {
        for (int query_index = 0; query_index < n_queries; ++query_index) {
            output.push_back({});
        }
        return output;
    }

    const int per_segment_k = std::max(segment_search_k, k);
    const int segment_beam_width = std::max(ef, per_segment_k);
    const bool use_euclidean = metric == "euclidean";

    struct PreparedSegment {
        const HnswSegmentView* segment = nullptr;
        std::vector<const CsrLayerView*> layer_by_number;
        SearchScratch scratch;
    };

    std::vector<PreparedSegment> prepared_segments;
    prepared_segments.reserve(static_cast<std::size_t>(n_segments));

    for (int segment_index = 0; segment_index < n_segments; ++segment_index) {
        const HnswSegmentView& segment = segments[segment_index];
        if (segment.dimension != dimension) {
            throw std::invalid_argument("segment dimension must match queries dimension");
        }
        if (
            segment.vectors == nullptr ||
            segment.layers == nullptr ||
            segment.n_vectors <= 0 ||
            segment.n_layers <= 0 ||
            segment.entry_point < 0 ||
            segment.entry_point >= segment.n_vectors ||
            segment.max_layer < 0
        ) {
            continue;
        }

        prepared_segments.push_back(PreparedSegment());
        PreparedSegment& prepared = prepared_segments.back();
        prepared.segment = &segment;
        prepared.layer_by_number.assign(
            static_cast<std::size_t>(segment.max_layer + 1),
            nullptr
        );
        prepared.scratch.ensure_size(static_cast<std::size_t>(segment.n_vectors));

        for (int layer_index = 0; layer_index < segment.n_layers; ++layer_index) {
            const CsrLayerView& layer = segment.layers[layer_index];
            if (layer.layer >= 0 && layer.layer <= segment.max_layer) {
                prepared.layer_by_number[static_cast<std::size_t>(layer.layer)] = &layer;
            }
        }
    }

    for (int query_index = 0; query_index < n_queries; ++query_index) {
        const float* query = queries + (static_cast<std::size_t>(query_index) * dimension);
        std::vector<SearchResult> merged;
        merged.reserve(
            static_cast<std::size_t>(prepared_segments.size()) *
            static_cast<std::size_t>(per_segment_k)
        );

        for (PreparedSegment& prepared : prepared_segments) {
            const HnswSegmentView& segment = *prepared.segment;
            std::vector<int> nearest;
            nearest.push_back(segment.entry_point);

            for (int layer = segment.max_layer; layer > 0; --layer) {
                const CsrLayerView* layer_view =
                    prepared.layer_by_number[static_cast<std::size_t>(layer)];
                if (layer_view == nullptr) {
                    continue;
                }

                const std::vector<SearchResult> candidates = search_csr_layer(
                    query,
                    segment.vectors,
                    segment.n_vectors,
                    segment.dimension,
                    layer_view->offsets,
                    layer_view->offsets_len,
                    layer_view->neighbors,
                    layer_view->neighbors_len,
                    nearest.data(),
                    static_cast<int>(nearest.size()),
                    1,
                    use_euclidean,
                    prepared.scratch
                );
                if (!candidates.empty()) {
                    nearest.clear();
                    nearest.push_back(candidates[0].id);
                }
            }

            const CsrLayerView* base_layer = prepared.layer_by_number.empty()
                ? nullptr
                : prepared.layer_by_number[0];
            if (base_layer == nullptr) {
                continue;
            }

            std::vector<SearchResult> candidates = search_csr_layer(
                query,
                segment.vectors,
                segment.n_vectors,
                segment.dimension,
                base_layer->offsets,
                base_layer->offsets_len,
                base_layer->neighbors,
                base_layer->neighbors_len,
                nearest.data(),
                static_cast<int>(nearest.size()),
                segment_beam_width,
                use_euclidean,
                prepared.scratch
            );
            if (static_cast<int>(candidates.size()) > per_segment_k) {
                candidates.resize(static_cast<std::size_t>(per_segment_k));
            }
            for (const SearchResult& candidate : candidates) {
                if (candidate.id < 0 || candidate.id >= segment.n_vectors) {
                    continue;
                }
                merged.push_back({
                    segment.global_offset + candidate.id,
                    candidate.distance
                });
            }
        }

        std::sort(
            merged.begin(),
            merged.end(),
            [use_euclidean](const SearchResult& left, const SearchResult& right) {
                return public_result_less(left, right, use_euclidean);
            }
        );
        if (static_cast<int>(merged.size()) > k) {
            merged.resize(static_cast<std::size_t>(k));
        }
        output.push_back(merged);
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
        connection_ids == nullptr ||
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
    const std::vector<int> connection_vector(
        connection_ids,
        connection_ids + static_cast<std::size_t>(n_connection_ids)
    );

    return prune_connection_vector(
        vectors,
        n_vectors,
        dimension,
        node_id,
        connection_vector,
        max_connections,
        use_euclidean,
        true,
        nullptr
    );
}

std::vector<int> select_heuristic_neighbors(
    const float* vectors,
    int n_vectors,
    int dimension,
    int node_id,
    const int* candidate_ids,
    int n_candidate_ids,
    int max_connections,
    const std::string& metric
) {
    if (metric != "euclidean" && metric != "cosine") {
        throw std::invalid_argument("metric must be 'euclidean' or 'cosine'");
    }
    if (
        vectors == nullptr ||
        candidate_ids == nullptr ||
        n_vectors <= 0 ||
        dimension <= 0 ||
        node_id < 0 ||
        node_id >= n_vectors ||
        n_candidate_ids <= 0 ||
        max_connections <= 0
    ) {
        return {};
    }

    const bool use_euclidean = metric == "euclidean";
    const float* node_vector = vectors + (static_cast<std::size_t>(node_id) * dimension);
    std::vector<HeapItem> candidates;
    candidates.reserve(static_cast<std::size_t>(n_candidate_ids));

    for (int i = 0; i < n_candidate_ids; ++i) {
        const int candidate_id = candidate_ids[i];
        if (candidate_id < 0 || candidate_id >= n_vectors || candidate_id == node_id) {
            continue;
        }

        const float* candidate_vector =
            vectors + (static_cast<std::size_t>(candidate_id) * dimension);
        const double distance = use_euclidean
            ? squared_euclidean_distance(node_vector, candidate_vector, dimension)
            : -cosine_similarity(node_vector, candidate_vector, dimension);
        candidates.push_back({distance, candidate_id});
    }

    std::sort(candidates.begin(), candidates.end(), [](const HeapItem& left, const HeapItem& right) {
        if (left.distance == right.distance) {
            return left.id < right.id;
        }
        return left.distance < right.distance;
    });

    return select_heuristic_neighbor_ids(
        vectors,
        n_vectors,
        dimension,
        node_id,
        candidates,
        max_connections,
        use_euclidean,
        nullptr
    );
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
    const Clock::time_point total_start = Clock::now();

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
        empty_result.build_stats.uses_squared_l2 = metric == "euclidean";
        empty_result.build_stats.uses_float_l2_accumulation = metric == "euclidean";
        empty_result.build_stats.uses_reusable_search_heaps = true;
        empty_result.build_stats.uses_bounded_adjacency = true;
        empty_result.build_stats.uses_heuristic_neighbors = true;
        empty_result.build_stats.uses_heuristic_reverse_pruning = false;
        empty_result.build_stats.total_seconds = seconds_since(total_start);
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

    const Clock::time_point construction_start = Clock::now();
    double search_seconds = 0.0;
    double greedy_search_seconds = 0.0;
    double candidate_search_seconds = 0.0;
    double prune_seconds = 0.0;
    int search_calls = 0;
    int greedy_search_calls = 0;
    int candidate_search_calls = 0;
    int adjacency_layers_allocated = 0;
    int max_observed_degree = 0;
    SearchScratch search_scratch;
    SearchInstrumentation search_instrumentation;
    DistanceInstrumentation distance_instrumentation;
    long long neighbor_selection_calls = 0;
    long long selected_degree_total = 0;
    int max_selected_degree = 0;
    long long prune_calls = 0;
    long long prune_input_total = 0;
    int max_prune_input_size = 0;
    search_scratch.ensure_size(static_cast<std::size_t>(n_vectors));

    for (int vector_id = 0; vector_id < n_vectors; ++vector_id) {
        const int node_level = levels[vector_id];
        if (node_level < 0) {
            throw std::invalid_argument("levels must be non-negative");
        }

        BuildNode node;
        node.level = node_level;
        for (int layer = 0; layer <= node_level; ++layer) {
            ensure_layer(node, layer, max_connections, adjacency_layers_allocated);
        }
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
            const Clock::time_point search_start = Clock::now();
            ++search_calls;
            ++greedy_search_calls;
            const std::vector<HeapItem> candidates = search_mutable_layer(
                query,
                vectors,
                n_vectors,
                dimension,
                nodes,
                nearest,
                1,
                layer,
                use_euclidean,
                search_scratch,
                &search_instrumentation
            );
            const double elapsed = seconds_since(search_start);
            search_seconds += elapsed;
            greedy_search_seconds += elapsed;
            std::vector<int> next_nearest = select_closest_ids(candidates, 1);
            if (!next_nearest.empty()) {
                nearest = next_nearest;
            }
        }

        for (int layer = node_level; layer >= 0; --layer) {
            const Clock::time_point search_start = Clock::now();
            ++search_calls;
            ++candidate_search_calls;
            const std::vector<HeapItem> candidates = search_mutable_layer(
                query,
                vectors,
                n_vectors,
                dimension,
                nodes,
                nearest,
                ef_construction,
                layer,
                use_euclidean,
                search_scratch,
                &search_instrumentation
            );
            const double elapsed = seconds_since(search_start);
            search_seconds += elapsed;
            candidate_search_seconds += elapsed;

            const int layer_max_connections = layer == 0 ? max_connections * 2 : max_connections;
            ++neighbor_selection_calls;
            std::vector<int> selected_neighbors = select_heuristic_neighbor_ids(
                vectors,
                n_vectors,
                dimension,
                vector_id,
                candidates,
                layer_max_connections,
                use_euclidean,
                &distance_instrumentation.neighbor_selection_distance_evaluations
            );
            selected_degree_total += static_cast<long long>(selected_neighbors.size());
            max_selected_degree = std::max(
                max_selected_degree,
                static_cast<int>(selected_neighbors.size())
            );

            ensure_layer(
                nodes[static_cast<std::size_t>(vector_id)],
                layer,
                max_connections,
                adjacency_layers_allocated
            );
            nodes[static_cast<std::size_t>(vector_id)]
                .connections[static_cast<std::size_t>(layer)]
                .neighbors = selected_neighbors;
            max_observed_degree = std::max(
                max_observed_degree,
                static_cast<int>(selected_neighbors.size())
            );

            for (int neighbor_id : selected_neighbors) {
                if (
                    neighbor_id < 0 ||
                    neighbor_id >= n_vectors ||
                    static_cast<std::size_t>(neighbor_id) >= nodes.size()
                ) {
                    continue;
                }

                BuildNode& neighbor_node = nodes[static_cast<std::size_t>(neighbor_id)];
                ensure_layer(
                    neighbor_node,
                    layer,
                    max_connections,
                    adjacency_layers_allocated
                );
                BuildLayer& neighbor_layer =
                    neighbor_node.connections[static_cast<std::size_t>(layer)];
                std::vector<int>& neighbor_connections = neighbor_layer.neighbors;
                add_unique_connection(neighbor_layer, vector_id);

                if (static_cast<int>(neighbor_connections.size()) > layer_max_connections) {
                    const Clock::time_point prune_start = Clock::now();
                    ++prune_calls;
                    prune_input_total += static_cast<long long>(neighbor_connections.size());
                    max_prune_input_size = std::max(
                        max_prune_input_size,
                        static_cast<int>(neighbor_connections.size())
                    );
                    neighbor_connections = prune_connection_vector(
                        vectors,
                        n_vectors,
                        dimension,
                        neighbor_id,
                        neighbor_connections,
                        layer_max_connections,
                        use_euclidean,
                        false,
                        &distance_instrumentation.prune_distance_evaluations
                    );
                    prune_seconds += seconds_since(prune_start);
                }
                max_observed_degree = std::max(
                    max_observed_degree,
                    static_cast<int>(neighbor_connections.size())
                );
            }

            nearest = selected_neighbors;
        }

        if (node_level > current_max_layer) {
            current_max_layer = node_level;
            entry_point = vector_id;
        }
    }
    const double construction_seconds = seconds_since(construction_start);

    result.entry_point = entry_point;
    result.max_layer = current_max_layer;

    if (include_connections) {
        for (std::size_t node_id = 0; node_id < nodes.size(); ++node_id) {
            const BuildNode& node = nodes[node_id];
            for (std::size_t layer = 0; layer < node.connections.size(); ++layer) {
                if (node.connections[layer].neighbors.empty()) {
                    continue;
                }

                LayerConnection connection;
                connection.node_id = static_cast<int>(node_id);
                connection.layer = static_cast<int>(layer);
                connection.neighbors = node.connections[layer].neighbors;
                std::sort(connection.neighbors.begin(), connection.neighbors.end());
                result.connections.push_back(connection);
            }
        }
    }

    const Clock::time_point csr_export_start = Clock::now();
    long long directed_edges = 0;
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

            std::vector<int> sorted_neighbors =
                node.connections[static_cast<std::size_t>(layer)].neighbors;
            directed_edges += static_cast<long long>(sorted_neighbors.size());
            max_observed_degree = std::max(
                max_observed_degree,
                static_cast<int>(sorted_neighbors.size())
            );
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
    const double csr_export_seconds = seconds_since(csr_export_start);

    result.build_stats.vectors = n_vectors;
    result.build_stats.dimensions = dimension;
    result.build_stats.max_layer = current_max_layer;
    result.build_stats.directed_edges = directed_edges;
    result.build_stats.construction_seconds = construction_seconds;
    result.build_stats.search_seconds = search_seconds;
    result.build_stats.greedy_search_seconds = greedy_search_seconds;
    result.build_stats.candidate_search_seconds = candidate_search_seconds;
    result.build_stats.prune_seconds = prune_seconds;
    result.build_stats.csr_export_seconds = csr_export_seconds;
    result.build_stats.uses_squared_l2 = use_euclidean;
    result.build_stats.uses_float_l2_accumulation = use_euclidean;
    result.build_stats.search_calls = search_calls;
    result.build_stats.greedy_search_calls = greedy_search_calls;
    result.build_stats.candidate_search_calls = candidate_search_calls;
    result.build_stats.visited_resizes = search_scratch.visited_resizes;
    result.build_stats.uses_reusable_search_heaps = true;
    result.build_stats.search_heap_resizes = search_scratch.search_heap_resizes;
    result.build_stats.uses_bounded_adjacency = true;
    result.build_stats.uses_heuristic_neighbors = true;
    result.build_stats.uses_heuristic_reverse_pruning = false;
    result.build_stats.adjacency_layers_allocated = adjacency_layers_allocated;
    result.build_stats.max_observed_degree = max_observed_degree;
    result.build_stats.search_distance_evaluations =
        search_instrumentation.distance_evaluations;
    result.build_stats.neighbor_selection_distance_evaluations =
        distance_instrumentation.neighbor_selection_distance_evaluations;
    result.build_stats.prune_distance_evaluations =
        distance_instrumentation.prune_distance_evaluations;
    result.build_stats.distance_evaluations =
        result.build_stats.search_distance_evaluations
        + result.build_stats.neighbor_selection_distance_evaluations
        + result.build_stats.prune_distance_evaluations;
    result.build_stats.visited_nodes = search_instrumentation.visited_nodes;
    result.build_stats.max_visited_nodes_per_search =
        search_instrumentation.max_visited_nodes_per_search;
    result.build_stats.candidate_heap_pushes =
        search_instrumentation.candidate_heap_pushes;
    result.build_stats.result_heap_pushes =
        search_instrumentation.result_heap_pushes;
    result.build_stats.neighbor_selection_calls = neighbor_selection_calls;
    result.build_stats.selected_degree_total = selected_degree_total;
    result.build_stats.average_selected_degree = neighbor_selection_calls == 0
        ? 0.0
        : static_cast<double>(selected_degree_total) / static_cast<double>(neighbor_selection_calls);
    result.build_stats.max_selected_degree = max_selected_degree;
    result.build_stats.prune_calls = prune_calls;
    result.build_stats.prune_input_total = prune_input_total;
    result.build_stats.average_prune_input_size = prune_calls == 0
        ? 0.0
        : static_cast<double>(prune_input_total) / static_cast<double>(prune_calls);
    result.build_stats.max_prune_input_size = max_prune_input_size;
    result.build_stats.total_seconds = seconds_since(total_start);

    return result;
}

}  // namespace vectordb
