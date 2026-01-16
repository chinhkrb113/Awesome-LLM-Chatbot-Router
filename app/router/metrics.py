import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class LatencyStats:
    """Rolling window latency statistics."""
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    avg: float = 0.0
    count: int = 0


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    embed_latency: LatencyStats
    search_latency: LatencyStats
    cache_hit_rate: float
    fallback_count: int
    reload_count: int
    error_count: int
    last_reload_duration_ms: float


class RouterMetrics:
    """
    Lightweight metrics collector for Router v2.
    """
    
    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._lock = threading.Lock()
        
        # Rolling windows for latency
        self._embed_latencies: deque = deque(maxlen=window_size)
        self._search_latencies: deque = deque(maxlen=window_size)
        
        # Counters
        self._cache_hits = 0
        self._cache_misses = 0
        self._fallback_count = 0
        self._reload_count = 0
        self._error_count = 0
        self._last_reload_duration_ms = 0.0
    
    def record_embed_latency(self, duration_ms: float):
        """Record embedding encode latency."""
        with self._lock:
            self._embed_latencies.append(duration_ms)
    
    def record_search_latency(self, duration_ms: float):
        """Record vector search latency."""
        with self._lock:
            self._search_latencies.append(duration_ms)
    
    def record_cache_hit(self):
        with self._lock:
            self._cache_hits += 1
    
    def record_cache_miss(self):
        with self._lock:
            self._cache_misses += 1
    
    def record_fallback_used(self):
        with self._lock:
            self._fallback_count += 1
    
    def record_reload(self, duration_ms: float):
        with self._lock:
            self._reload_count += 1
            self._last_reload_duration_ms = duration_ms
    
    def record_error(self):
        with self._lock:
            self._error_count += 1
    
    def get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        with self._lock:
            embed_stats = self._compute_latency_stats(list(self._embed_latencies))
            search_stats = self._compute_latency_stats(list(self._search_latencies))
            
            total_cache = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / total_cache if total_cache > 0 else 0.0
            
            return MetricsSnapshot(
                embed_latency=embed_stats,
                search_latency=search_stats,
                cache_hit_rate=round(cache_hit_rate, 4),
                fallback_count=self._fallback_count,
                reload_count=self._reload_count,
                error_count=self._error_count,
                last_reload_duration_ms=self._last_reload_duration_ms
            )
    
    def _compute_latency_stats(self, values: List[float]) -> LatencyStats:
        if not values:
            return LatencyStats()
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return LatencyStats(
            p50=sorted_vals[int(n * 0.50)] if n > 0 else 0,
            p95=sorted_vals[int(n * 0.95)] if n > 0 else 0,
            p99=sorted_vals[int(n * 0.99)] if n > 0 else 0,
            avg=round(statistics.mean(values), 2),
            count=n
        )
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        snapshot = self.get_snapshot()
        lines = [
            f"# HELP router_embed_latency_ms Embedding encode latency",
            f"# TYPE router_embed_latency_ms summary",
            f'router_embed_latency_ms{{quantile="0.5"}} {snapshot.embed_latency.p50}',
            f'router_embed_latency_ms{{quantile="0.95"}} {snapshot.embed_latency.p95}',
            f'router_embed_latency_ms{{quantile="0.99"}} {snapshot.embed_latency.p99}',
            f"router_embed_latency_ms_count {snapshot.embed_latency.count}",
            f"",
            f"# HELP router_search_latency_ms Vector search latency",
            f"# TYPE router_search_latency_ms summary",
            f'router_search_latency_ms{{quantile="0.5"}} {snapshot.search_latency.p50}',
            f'router_search_latency_ms{{quantile="0.95"}} {snapshot.search_latency.p95}',
            f'router_search_latency_ms{{quantile="0.99"}} {snapshot.search_latency.p99}',
            f"router_search_latency_ms_count {snapshot.search_latency.count}",
            f"",
            f"# HELP router_cache_hit_rate Cache hit rate",
            f"# TYPE router_cache_hit_rate gauge",
            f"router_cache_hit_rate {snapshot.cache_hit_rate}",
            f"",
            f"# HELP router_fallback_total Fallback model usage count",
            f"# TYPE router_fallback_total counter",
            f"router_fallback_total {snapshot.fallback_count}",
            f"",
            f"# HELP router_reload_total Config reload count",
            f"# TYPE router_reload_total counter",
            f"router_reload_total {snapshot.reload_count}",
            f"",
            f"# HELP router_error_total Error count",
            f"# TYPE router_error_total counter",
            f"router_error_total {snapshot.error_count}",
        ]
        return "\n".join(lines)


# Global metrics instance
_metrics: Optional[RouterMetrics] = None

def get_metrics() -> RouterMetrics:
    global _metrics
    if _metrics is None:
        _metrics = RouterMetrics()
    return _metrics


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, callback):
        self._callback = callback
        self._start = None
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        duration_ms = (time.perf_counter() - self._start) * 1000
        self._callback(duration_ms)


def time_embed():
    """Timer for embedding operations."""
    return TimerContext(get_metrics().record_embed_latency)

def time_search():
    """Timer for search operations."""
    return TimerContext(get_metrics().record_search_latency)
