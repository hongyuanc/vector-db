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

    cdef cppclass CppBuildGraphResult "vectordb::BuildGraphResult":
        int entry_point
        int max_layer
        vector[int] levels
        vector[CppLayerConnection] connections
        vector[CppCsrLayer] layers

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
    }
