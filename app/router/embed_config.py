from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict

class VectorStoreType(Enum):
    MEMORY = "memory"
    FAISS = "faiss"
    QDRANT = "qdrant"

class ModelType(Enum):
    JINA_V3 = "jina-v3"
    BGE_M3 = "bge-m3"
    VIETNAMESE_SBERT = "vietnamese-sbert"

# Model dimension mapping (source of truth)
MODEL_DIMENSIONS: Dict[str, int] = {
    "jinaai/jina-embeddings-v3": 1024,
    "BAAI/bge-m3": 1024,
    "keepitreal/vietnamese-sbert": 768,
    # Matryoshka dimensions for Jina v3
    "jinaai/jina-embeddings-v3:256": 256,
    "jinaai/jina-embeddings-v3:512": 512,
    "jinaai/jina-embeddings-v3:768": 768,
    "jinaai/jina-embeddings-v3:1024": 1024,
}

@dataclass
class EmbedConfig:
    # Model settings
    # Jina v3 is currently not supported by embed-anything v0.7.0
    # Using BGE-Small as a stable default
    primary_model: str = "BAAI/bge-small-en-v1.5"
    fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Matryoshka dimension (None = use default)
    # Jina v3 supports: 256, 512, 768, 1024
    matryoshka_dim: Optional[int] = None
    
    # Task type for Jina v3 (retrieval.query vs retrieval.passage)
    task_type: str = "retrieval.query"
    
    # Vector store
    vector_store: VectorStoreType = VectorStoreType.MEMORY
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "action_seeds"
    
    # Scoring strategy
    max_pool: bool = True
    
    # Cache settings
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 300
    
    # Hot reload
    reload_poll_interval: float = 2.0
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension based on model and matryoshka setting."""
        if self.matryoshka_dim:
            return self.matryoshka_dim
        return MODEL_DIMENSIONS.get(self.primary_model, 1024)
    
    @property
    def fallback_dim(self) -> int:
        return MODEL_DIMENSIONS.get(self.fallback_model, 1024)
