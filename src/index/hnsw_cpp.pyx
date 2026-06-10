# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Cython wrapper for the C++ HNSW search primitives.
"""

import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cnp.import_array()


cdef extern from "hnsw_cpp_core.hpp" namespace "vectordb":
    cdef cppclass CppSearchResult "vectordb::SearchResult":
        int id
        double distance

    cdef cppclass CppLayerConnection "vectordb::LayerConnection":
        int node_id
        int layer
        vector[int] neighbors

    cdef cppclass CppCsrLayer "vectordb::CsrLayer":
        int layer
        vector[int] offsets
        vector[int] neighbors

    cdef cppclass CppCsrLayerView "vectordb::CsrLayerView":
        int layer
        const int* offsets
        int offsets_len
        const int* neighbors
        int neighbors_len

    cdef cppclass CppBuildStats "vectordb::BuildStats":
        int vectors
        int dimensions
        int max_layer
        long long directed_edges
        double total_seconds
        double construction_seconds
        double search_seconds
        double greedy_search_seconds
        double candidate_search_seconds
        double prune_seconds
        double csr_export_seconds
        bool uses_squared_l2
        bool uses_float_l2_accumulation
        int search_calls
        int greedy_search_calls
        int candidate_search_calls
        int visited_resizes
        bool uses_reusable_search_heaps
        int search_heap_resizes
        bool uses_bounded_adjacency
        bool uses_heuristic_neighbors
        bool uses_heuristic_reverse_pruning
        int adjacency_layers_allocated
        int max_observed_degree
        long long distance_evaluations
        long long search_distance_evaluations
        long long neighbor_selection_distance_evaluations
        long long prune_distance_evaluations
        long long visited_nodes
        int max_visited_nodes_per_search
        long long candidate_heap_pushes
        long long result_heap_pushes
        long long neighbor_selection_calls
        long long selected_degree_total
        double average_selected_degree
        int max_selected_degree
        long long prune_calls
        long long prune_input_total
        double average_prune_input_size
        int max_prune_input_size

    cdef cppclass CppBuildGraphResult "vectordb::BuildGraphResult":
        int entry_point
        int max_layer
        vector[int] levels
        vector[CppLayerConnection] connections
        vector[CppCsrLayer] layers
        CppBuildStats build_stats

    vector[CppSearchResult] cpp_search_layer "vectordb::search_layer"(
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
        const string& metric
    ) except +

    vector[vector[CppSearchResult]] cpp_search_batch "vectordb::search_batch"(
        const float* queries,
        int n_queries,
        const float* vectors,
        int n_vectors,
        int dimension,
        const CppCsrLayerView* layers,
        int n_layers,
        int entry_point,
        int max_layer,
        int k,
        int ef,
        const string& metric
    ) except +

    vector[int] cpp_prune_connections "vectordb::prune_connections"(
        const float* vectors,
        int n_vectors,
        int dimension,
        int node_id,
        const int* connection_ids,
        int n_connection_ids,
        int max_connections,
        const string& metric
    ) except +

    vector[int] cpp_select_heuristic_neighbors "vectordb::select_heuristic_neighbors"(
        const float* vectors,
        int n_vectors,
        int dimension,
        int node_id,
        const int* candidate_ids,
        int n_candidate_ids,
        int max_connections,
        const string& metric
    ) except +

    CppBuildGraphResult cpp_build_graph "vectordb::build_graph"(
        const float* vectors,
        int n_vectors,
        int dimension,
        const int* levels,
        int n_levels,
        int max_connections,
        int ef_construction,
        const string& metric,
        bool include_connections
    ) except +


def search_layer(
    query,
    vectors,
    offsets,
    neighbors,
    entry_points,
    int num_closest,
    str metric,
):
    """
    Search one HNSW graph layer using C++ heaps and CSR adjacency arrays.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] query_arr = np.ascontiguousarray(
        query, dtype=np.float32
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors_arr = np.ascontiguousarray(
        vectors, dtype=np.float32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] offsets_arr = np.ascontiguousarray(
        offsets, dtype=np.int32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] neighbors_arr = np.ascontiguousarray(
        neighbors, dtype=np.int32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] entry_arr = np.ascontiguousarray(
        entry_points, dtype=np.int32
    )
    cdef string metric_cpp = metric.encode("utf-8")

    cdef vector[CppSearchResult] raw = cpp_search_layer(
        <const float*> query_arr.data,
        <const float*> vectors_arr.data,
        <int> vectors_arr.shape[0],
        <int> vectors_arr.shape[1],
        <const int*> offsets_arr.data,
        <int> offsets_arr.shape[0],
        <const int*> neighbors_arr.data,
        <int> neighbors_arr.shape[0],
        <const int*> entry_arr.data,
        <int> entry_arr.shape[0],
        num_closest,
        metric_cpp,
    )

    cdef Py_ssize_t i
    cdef list results = []
    for i in range(raw.size()):
        results.append((raw[i].id, raw[i].distance))

    return results


def search_batch(
    queries,
    vectors,
    layers,
    int entry_point,
    int max_layer,
    int k,
    int ef,
    str metric,
):
    """
    Search a compact CSR HNSW index for a batch of query vectors.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] queries_arr = np.ascontiguousarray(
        queries, dtype=np.float32
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors_arr = np.ascontiguousarray(
        vectors, dtype=np.float32
    )
    if queries_arr.shape[1] != vectors_arr.shape[1]:
        raise ValueError("queries dimension must match vectors dimension")

    cdef vector[CppCsrLayerView] layer_views
    cdef CppCsrLayerView layer_view
    cdef list layer_arrays = []
    cdef object layer
    cdef object layer_data
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] offsets_arr
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] neighbors_arr

    for layer in sorted(layers):
        layer_data = layers[layer]
        offsets_arr = np.ascontiguousarray(layer_data[0], dtype=np.int32)
        neighbors_arr = np.ascontiguousarray(layer_data[1], dtype=np.int32)
        layer_arrays.append((offsets_arr, neighbors_arr))

        layer_view.layer = <int> layer
        layer_view.offsets = <const int*> offsets_arr.data
        layer_view.offsets_len = <int> offsets_arr.shape[0]
        layer_view.neighbors = <const int*> neighbors_arr.data
        layer_view.neighbors_len = <int> neighbors_arr.shape[0]
        layer_views.push_back(layer_view)

    cdef const CppCsrLayerView* layer_ptr = NULL
    if layer_views.size() > 0:
        layer_ptr = &layer_views[0]

    cdef string metric_cpp = metric.encode("utf-8")
    cdef vector[vector[CppSearchResult]] raw = cpp_search_batch(
        <const float*> queries_arr.data,
        <int> queries_arr.shape[0],
        <const float*> vectors_arr.data,
        <int> vectors_arr.shape[0],
        <int> vectors_arr.shape[1],
        layer_ptr,
        <int> layer_views.size(),
        entry_point,
        max_layer,
        k,
        ef,
        metric_cpp,
    )

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef list output = []
    cdef list query_results
    for i in range(raw.size()):
        query_results = []
        for j in range(raw[i].size()):
            query_results.append((raw[i][j].id, raw[i][j].distance))
        output.append(query_results)

    return output


def prune_connections(
    vectors,
    int node_id,
    connection_ids,
    int max_connections,
    str metric,
):
    """
    Prune one node's connections using C++ distance calculation and sorting.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors_arr = np.ascontiguousarray(
        vectors, dtype=np.float32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] connection_arr = np.ascontiguousarray(
        connection_ids, dtype=np.int32
    )
    cdef string metric_cpp = metric.encode("utf-8")

    cdef vector[int] raw = cpp_prune_connections(
        <const float*> vectors_arr.data,
        <int> vectors_arr.shape[0],
        <int> vectors_arr.shape[1],
        node_id,
        <const int*> connection_arr.data,
        <int> connection_arr.shape[0],
        max_connections,
        metric_cpp,
    )

    cdef Py_ssize_t i
    cdef list results = []
    for i in range(raw.size()):
        results.append(raw[i])

    return results


def select_heuristic_neighbors(
    vectors,
    int node_id,
    candidate_ids,
    int max_connections,
    str metric,
):
    """
    Select neighbors with the native HNSW diversity heuristic.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors_arr = np.ascontiguousarray(
        vectors, dtype=np.float32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] candidate_arr = np.ascontiguousarray(
        candidate_ids, dtype=np.int32
    )
    cdef string metric_cpp = metric.encode("utf-8")

    cdef vector[int] raw = cpp_select_heuristic_neighbors(
        <const float*> vectors_arr.data,
        <int> vectors_arr.shape[0],
        <int> vectors_arr.shape[1],
        node_id,
        <const int*> candidate_arr.data,
        <int> candidate_arr.shape[0],
        max_connections,
        metric_cpp,
    )

    cdef Py_ssize_t i
    cdef list results = []
    for i in range(raw.size()):
        results.append(raw[i])

    return results


def build_graph(
    vectors,
    levels,
    int max_connections,
    int ef_construction,
    str metric,
    bint include_connections=True,
):
    """
    Build a complete HNSW graph in C++ and return Python-friendly graph rows.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] vectors_arr = np.ascontiguousarray(
        vectors, dtype=np.float32
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] levels_arr = np.ascontiguousarray(
        levels, dtype=np.int32
    )
    if levels_arr.shape[0] != vectors_arr.shape[0]:
        raise ValueError("levels length must match number of vectors")

    cdef string metric_cpp = metric.encode("utf-8")
    cdef CppBuildGraphResult raw = cpp_build_graph(
        <const float*> vectors_arr.data,
        <int> vectors_arr.shape[0],
        <int> vectors_arr.shape[1],
        <const int*> levels_arr.data,
        <int> levels_arr.shape[0],
        max_connections,
        ef_construction,
        metric_cpp,
        include_connections,
    )

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef CppLayerConnection connection
    cdef CppCsrLayer csr_layer
    cdef list output_levels = []
    cdef list output_connections = []
    cdef dict output_layers = {}
    cdef list neighbors
    cdef list offsets

    for i in range(raw.levels.size()):
        output_levels.append(raw.levels[i])

    if include_connections:
        for i in range(raw.connections.size()):
            connection = raw.connections[i]
            neighbors = []
            for j in range(connection.neighbors.size()):
                neighbors.append(connection.neighbors[j])
            output_connections.append((connection.node_id, connection.layer, neighbors))

    for i in range(raw.layers.size()):
        csr_layer = raw.layers[i]
        offsets = []
        neighbors = []
        for j in range(csr_layer.offsets.size()):
            offsets.append(csr_layer.offsets[j])
        for j in range(csr_layer.neighbors.size()):
            neighbors.append(csr_layer.neighbors[j])
        output_layers[csr_layer.layer] = {
            "offsets": np.asarray(offsets, dtype=np.int32),
            "neighbors": np.asarray(neighbors, dtype=np.int32),
        }

    return {
        "entry_point": raw.entry_point,
        "max_layer": raw.max_layer,
        "levels": output_levels,
        "connections": output_connections,
        "layers": output_layers,
        "build_stats": {
            "vectors": raw.build_stats.vectors,
            "dimensions": raw.build_stats.dimensions,
            "max_layer": raw.build_stats.max_layer,
            "directed_edges": raw.build_stats.directed_edges,
            "total_seconds": raw.build_stats.total_seconds,
            "construction_seconds": raw.build_stats.construction_seconds,
            "search_seconds": raw.build_stats.search_seconds,
            "greedy_search_seconds": raw.build_stats.greedy_search_seconds,
            "candidate_search_seconds": raw.build_stats.candidate_search_seconds,
            "prune_seconds": raw.build_stats.prune_seconds,
            "csr_export_seconds": raw.build_stats.csr_export_seconds,
            "uses_squared_l2": True if raw.build_stats.uses_squared_l2 else False,
            "uses_float_l2_accumulation": True if raw.build_stats.uses_float_l2_accumulation else False,
            "search_calls": raw.build_stats.search_calls,
            "greedy_search_calls": raw.build_stats.greedy_search_calls,
            "candidate_search_calls": raw.build_stats.candidate_search_calls,
            "visited_resizes": raw.build_stats.visited_resizes,
            "uses_reusable_search_heaps": True if raw.build_stats.uses_reusable_search_heaps else False,
            "search_heap_resizes": raw.build_stats.search_heap_resizes,
            "uses_bounded_adjacency": True if raw.build_stats.uses_bounded_adjacency else False,
            "uses_heuristic_neighbors": True if raw.build_stats.uses_heuristic_neighbors else False,
            "uses_heuristic_reverse_pruning": True if raw.build_stats.uses_heuristic_reverse_pruning else False,
            "adjacency_layers_allocated": raw.build_stats.adjacency_layers_allocated,
            "max_observed_degree": raw.build_stats.max_observed_degree,
            "distance_evaluations": raw.build_stats.distance_evaluations,
            "search_distance_evaluations": raw.build_stats.search_distance_evaluations,
            "neighbor_selection_distance_evaluations": raw.build_stats.neighbor_selection_distance_evaluations,
            "prune_distance_evaluations": raw.build_stats.prune_distance_evaluations,
            "visited_nodes": raw.build_stats.visited_nodes,
            "max_visited_nodes_per_search": raw.build_stats.max_visited_nodes_per_search,
            "candidate_heap_pushes": raw.build_stats.candidate_heap_pushes,
            "result_heap_pushes": raw.build_stats.result_heap_pushes,
            "neighbor_selection_calls": raw.build_stats.neighbor_selection_calls,
            "selected_degree_total": raw.build_stats.selected_degree_total,
            "average_selected_degree": raw.build_stats.average_selected_degree,
            "max_selected_degree": raw.build_stats.max_selected_degree,
            "prune_calls": raw.build_stats.prune_calls,
            "prune_input_total": raw.build_stats.prune_input_total,
            "average_prune_input_size": raw.build_stats.average_prune_input_size,
            "max_prune_input_size": raw.build_stats.max_prune_input_size,
        },
    }
