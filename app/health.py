from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(tags=["health"])

# Global references (set by main.py)
_router_instance = None
_embedding_engine = None

def set_instances(router_inst, embed_engine):
    global _router_instance, _embedding_engine
    _router_instance = router_inst
    _embedding_engine = embed_engine


@router.get("/healthz")
def health_check() -> Dict[str, Any]:
    """
    Liveness probe - process is running.
    """
    return {"status": "ok"}


@router.get("/readyz")
def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe - all components ready to serve.
    """
    checks = {
        "config_loaded": False,
        "embedding_ready": False,
        "vector_store_ready": False,
    }
    
    try:
        if _router_instance:
            checks["config_loaded"] = len(_router_instance.actions) > 0
        
        if _embedding_engine:
            stats = _embedding_engine.get_stats()
            checks["embedding_ready"] = stats.get("is_ready", False)
            checks["vector_store_ready"] = True  # If no error, it's ready
            
    except Exception as e:
        return {
            "status": "not_ready",
            "checks": checks,
            "error": str(e)
        }
    
    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    }


@router.get("/stats")
def get_stats() -> Dict[str, Any]:
    """
    Get detailed statistics.
    """
    stats = {}
    
    if _router_instance:
        stats["router"] = _router_instance.get_stats()
    
    if _embedding_engine:
        stats["embedding"] = _embedding_engine.get_stats()
    
    return stats
