"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class InsertRequest(BaseModel):
    """Request model for inserting a vector."""
    vector: List[float] = Field(..., description="Vector to insert")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata for the vector")


class InsertResponse(BaseModel):
    """Response model for insert operation."""
    id: int = Field(..., description="ID of the inserted vector")
    success: bool = Field(default=True, description="Whether the operation succeeded")


class SearchRequest(BaseModel):
    """Request model for searching vectors."""
    vector: List[float] = Field(..., description="Query vector")
    k: int = Field(default=10, ge=1, le=100, description="Number of neighbors to return")
    ef: Optional[int] = Field(default=None, description="Search width (HNSW parameter)")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class SearchResult(BaseModel):
    """Single search result."""
    id: int = Field(..., description="Vector ID")
    score: float = Field(..., description="Distance/similarity score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Vector metadata")


class SearchResponse(BaseModel):
    """Response model for search operation."""
    results: List[SearchResult] = Field(..., description="List of search results")
    latency_ms: float = Field(..., description="Query latency in milliseconds")


class BatchInsertRequest(BaseModel):
    """Request model for batch insert."""
    vectors: List[List[float]] = Field(..., description="List of vectors to insert")
    metadata: Optional[List[Dict[str, Any]]] = Field(default=None, description="Metadata for each vector")


class BatchInsertResponse(BaseModel):
    """Response model for batch insert."""
    ids: List[int] = Field(..., description="IDs of inserted vectors")
    count: int = Field(..., description="Number of vectors inserted")


class DeleteRequest(BaseModel):
    """Request model for deleting a vector."""
    id: int = Field(..., description="ID of the vector to delete")


class DeleteResponse(BaseModel):
    """Response model for delete operation."""
    success: bool = Field(default=True, description="Whether the operation succeeded")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(..., description="API version")
    vectors_count: int = Field(..., description="Number of vectors in database")
