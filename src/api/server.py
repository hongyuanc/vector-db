"""
FastAPI server for vector database operations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    InsertRequest,
    InsertResponse,
    SearchRequest,
    SearchResponse,
    BatchInsertRequest,
    BatchInsertResponse,
    DeleteRequest,
    DeleteResponse,
    HealthResponse,
)
import time
import numpy as np
from typing import Dict
from ..collection import Collection

# Initialize FastAPI app
app = FastAPI(
    title="Vector DB API",
    description="Production-grade vector database with HNSW indexing",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global collection manager
# For now, we'll use a single default collection
# TODO: Add multi-collection support
collections: Dict[str, Collection] = {}


def get_default_collection() -> Collection:
    """Get or create the default collection."""
    if "default" not in collections:
        raise HTTPException(
            status_code=503,
            detail="No collection initialized. Use POST /collections to create one."
        )
    return collections["default"]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vector_count = 0
    if "default" in collections:
        vector_count = collections["default"].count

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        vectors_count=vector_count,
    )


@app.post("/insert", response_model=InsertResponse)
async def insert_vector(request: InsertRequest):
    """
    Insert a single vector into the database.

    Args:
        request: Insert request with vector and optional metadata

    Returns:
        Insert response with vector ID
    """
    collection = get_default_collection()

    try:
        # Convert to numpy array
        vector = np.array(request.vector, dtype=np.float32)

        # Insert into collection
        vector_id = collection.insert(vector, metadata=request.metadata)

        return InsertResponse(id=vector_id, success=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """
    Search for k nearest neighbors.

    Args:
        request: Search request with query vector and parameters

    Returns:
        Search response with results and latency
    """
    collection = get_default_collection()

    start_time = time.time()

    try:
        # Convert to numpy array
        query = np.array(request.vector, dtype=np.float32)

        # Search
        results = collection.search(
            query=query,
            k=request.k,
            ef=request.ef,
            filter=request.filter
        )

        # Convert to response format
        from .models import SearchResult
        search_results = [
            SearchResult(id=vid, score=score, metadata=None)
            for vid, score in results
        ]

        latency_ms = (time.time() - start_time) * 1000

        return SearchResponse(results=search_results, latency_ms=latency_ms)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/batch_insert", response_model=BatchInsertResponse)
async def batch_insert_vectors(request: BatchInsertRequest):
    """
    Insert multiple vectors in a batch.

    Args:
        request: Batch insert request with vectors and metadata

    Returns:
        Batch insert response with vector IDs
    """
    collection = get_default_collection()

    try:
        # Convert to numpy array
        vectors = np.array(request.vectors, dtype=np.float32)

        # Batch insert
        vector_ids = collection.batch_insert(vectors, metadata=request.metadata)

        return BatchInsertResponse(ids=vector_ids, count=len(vector_ids))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch insert failed: {str(e)}")


@app.delete("/vector/{vector_id}", response_model=DeleteResponse)
async def delete_vector(vector_id: int):
    """
    Delete a vector by ID.

    Args:
        vector_id: ID of the vector to delete

    Returns:
        Delete response
    """
    # TODO: Implement delete logic (not yet supported in VectorStore/HNSW)
    raise HTTPException(status_code=501, detail="Delete operation not yet implemented")


@app.post("/collections/create")
async def create_collection(
    name: str = "default",
    dimension: int = 128,
    max_vectors: int = 1_000_000,
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    metric: str = "euclidean"
):
    """
    Create a new collection.

    Args:
        name: Collection name
        dimension: Vector dimension
        max_vectors: Maximum number of vectors
        M: HNSW M parameter
        ef_construction: HNSW ef_construction parameter
        ef_search: Default HNSW ef_search parameter
        metric: Distance metric
    """
    if name in collections:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{name}' already exists"
        )

    try:
        collection = Collection(
            name=name,
            dimension=dimension,
            max_vectors=max_vectors,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            metric=metric
        )
        collections[name] = collection

        return {
            "success": True,
            "message": f"Collection '{name}' created",
            "config": {
                "name": name,
                "dimension": dimension,
                "max_vectors": max_vectors,
                "M": M,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "metric": metric
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection: {str(e)}"
        )


@app.post("/collections/{name}/build_index")
async def build_index(name: str = "default"):
    """Build the HNSW index for a collection."""
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    collection = collections[name]

    try:
        collection.build_index()
        return {
            "success": True,
            "message": f"Index built for {collection.count} vectors"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    print("Starting Vector DB API...")
    print("Server ready at http://0.0.0.0:8000")
    print("Docs available at http://0.0.0.0:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down Vector DB API...")

    # Save all collections
    for name, collection in collections.items():
        print(f"Saving collection '{name}'...")
        collection.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
