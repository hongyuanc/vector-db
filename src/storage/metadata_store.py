"""
Metadata store for vector database using SQLite.

Stores metadata associated with vectors and enables filtering.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


class MetadataStore:
    """
    SQLite-based metadata storage for vectors.

    Supports:
    - Storing JSON metadata per vector
    - Filtering by metadata attributes
    - Lazy deletion with tombstones
    """

    def __init__(self, db_path: str):
        """
        Initialize metadata store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create schema
        self._create_schema()

    def _create_schema(self):
        """Create database schema."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    vector_id INTEGER PRIMARY KEY,
                    metadata TEXT,  -- JSON string
                    deleted BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deleted
                ON vector_metadata(deleted)
            """)

    def insert(self, vector_id: int, metadata: Optional[Dict[str, Any]] = None):
        """
        Insert metadata for a vector.

        Args:
            vector_id: Vector ID
            metadata: Metadata dictionary (will be stored as JSON)
        """
        metadata_json = json.dumps(metadata) if metadata else None

        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO vector_metadata (vector_id, metadata, deleted)
                VALUES (?, ?, 0)
            """, (vector_id, metadata_json))

    def get(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector.

        Args:
            vector_id: Vector ID

        Returns:
            Metadata dictionary or None if not found
        """
        cursor = self.conn.execute("""
            SELECT metadata FROM vector_metadata
            WHERE vector_id = ? AND deleted = 0
        """, (vector_id,))

        row = cursor.fetchone()
        if row and row['metadata']:
            return json.loads(row['metadata'])
        return None

    def delete(self, vector_id: int):
        """
        Mark a vector as deleted (lazy deletion).

        Args:
            vector_id: Vector ID to delete
        """
        with self.conn:
            self.conn.execute("""
                UPDATE vector_metadata
                SET deleted = 1, updated_at = CURRENT_TIMESTAMP
                WHERE vector_id = ?
            """, (vector_id,))

    def is_deleted(self, vector_id: int) -> bool:
        """
        Check if a vector is deleted.

        Args:
            vector_id: Vector ID

        Returns:
            True if deleted, False otherwise
        """
        cursor = self.conn.execute("""
            SELECT deleted FROM vector_metadata
            WHERE vector_id = ?
        """, (vector_id,))

        row = cursor.fetchone()
        return row['deleted'] == 1 if row else False

    def filter(self, conditions: Dict[str, Any]) -> List[int]:
        """
        Filter vectors by metadata conditions.

        Args:
            conditions: Dictionary of field -> value conditions

        Returns:
            List of vector IDs matching the conditions
        """
        # Simple equality filtering using JSON extraction
        # For production, consider using JSON1 extension or indexed columns

        query = """
            SELECT vector_id, metadata FROM vector_metadata
            WHERE deleted = 0
        """

        cursor = self.conn.execute(query)

        matching_ids = []
        for row in cursor:
            if row['metadata']:
                metadata = json.loads(row['metadata'])

                # Check if all conditions match
                match = True
                for key, value in conditions.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break

                if match:
                    matching_ids.append(row['vector_id'])

        return matching_ids

    def batch_get(self, vector_ids: List[int]) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Get metadata for multiple vectors.

        Args:
            vector_ids: List of vector IDs

        Returns:
            Dictionary mapping vector_id -> metadata
        """
        if not vector_ids:
            return {}

        placeholders = ','.join('?' * len(vector_ids))
        cursor = self.conn.execute(f"""
            SELECT vector_id, metadata FROM vector_metadata
            WHERE vector_id IN ({placeholders}) AND deleted = 0
        """, vector_ids)

        result = {}
        for row in cursor:
            metadata = json.loads(row['metadata']) if row['metadata'] else None
            result[row['vector_id']] = metadata

        return result

    def count(self) -> int:
        """
        Count non-deleted vectors.

        Returns:
            Number of active vectors
        """
        cursor = self.conn.execute("""
            SELECT COUNT(*) as count FROM vector_metadata
            WHERE deleted = 0
        """)

        return cursor.fetchone()['count']

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
