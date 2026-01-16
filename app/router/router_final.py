import os
import uuid
import time
import asyncio
import datetime
import threading
from typing import List, Optional, Dict

from app.core.models import (
    RouterOutput, UserRequest, ActionCandidate,
    UIStrategy, ActionConfig, ConversationContext, RouterTrace
)
from app.core.exceptions import ProcessingError, ContextError
from app.core.logger import system_logger as logger
from app.utils.config_loader import ConfigLoader
from app.router.preprocess import Preprocessor
from app.router.rule_engine import RuleEngine
from app.router.ui_decision import UIDecision
from app.router.embed_config import EmbedConfig
from app.router.embed_anything_engine_final import EmbedAnythingEngineFinal
from app.utils.config_watcher_v2 import AtomicConfigWatcher
from app.router.metrics import get_metrics, time_embed
from app.router.pairwise_resolver import PairwiseResolver
from app.action_flow.entity_extractor import EntitySignalExtractor

class FuserFinal:
    """Dynamic fuser with domain/intent-aware weights."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.loader = config_loader
        self.weights = self.loader.get_weights()
        
        # Keep domain weights logic as is, or move to config if needed
        self.domain_weights = {
            "leave": {"w_rule": 0.7, "w_embed": 0.3},
            "visitor": {"w_rule": 0.5, "w_embed": 0.5},
        }
        
        self.intent_adjustments = {
            "cancel": {"w_rule": 0.8, "w_embed": 0.2},
            "status": {"w_rule": 0.5, "w_embed": 0.5},
            "create": {"w_rule": 0.6, "w_embed": 0.4},
        }
    
    def fuse(self, rule_score: float, embed_score: float,
             domain: str = None, intent_type: str = None,
             has_pattern: bool = False,
             context_boost: float = 0.0,
             entity_boost: float = 0.0) -> float:
        
        # Use config defaults
        w_rule = self.weights.get("default_rule", 0.6)
        w_embed = self.weights.get("default_embed", 0.4)
        
        if domain and domain in self.domain_weights:
            w_rule = self.domain_weights[domain]["w_rule"]
            w_embed = self.domain_weights[domain]["w_embed"]
        
        if intent_type and intent_type in self.intent_adjustments:
            w_rule = self.intent_adjustments[intent_type]["w_rule"]
            w_embed = self.intent_adjustments[intent_type]["w_embed"]
        
        bonus = 0.15 if has_pattern else 0.0
        
        base = (w_rule * rule_score) + (w_embed * embed_score)
        final = base + bonus + context_boost + entity_boost
        
        return max(0.0, min(1.0, final))


class RouterFinal:
    """
    Production-final Router with all P0 + P0.5 fixes.
    Thread-safe implementation using RLock.
    """
    
    RESET_KEYWORDS = ["thôi", "huỷ", "hủy", "cancel", "bỏ", "không", "quên đi"]

    def __init__(self,
                 config_loader: ConfigLoader,
                 embed_config: EmbedConfig = None,
                 enable_v2: bool = True):
        
        self.loader = config_loader
        self.enable_v2 = enable_v2
        self.metrics = get_metrics()
        
        # Context Storage with Thread Safety
        self._contexts: Dict[str, ConversationContext] = {}
        self._lock = threading.RLock() # Reentrant lock for thread safety
        
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = True

        # Core components
        self.preprocessor = Preprocessor()
        self.rule_engine = RuleEngine()
        self.ui_decision = UIDecision()
        self.fuser = FuserFinal(config_loader) # Pass loader
        self.pairwise_resolver = PairwiseResolver(config_loader)
        self.entity_extractor = EntitySignalExtractor()
        
        # Embedding engine
        self.embed_config = embed_config or EmbedConfig()
        self.embedding_engine = EmbedAnythingEngineFinal(self.embed_config)
        
        self.actions: List[ActionConfig] = []
        
        # Initialize
        # self._initialize() -> Moved to async initialize()
        
    async def initialize(self):
        start = time.perf_counter()
        try:
            await self.loader.load()
            await self.pairwise_resolver.initialize()
            self.actions = self.loader.get_all_actions()
            self.embedding_engine.initialize(self.actions)
            
            duration_ms = (time.perf_counter() - start) * 1000
            self.metrics.record_reload(duration_ms)
            logger.info(f"Router initialized: {len(self.actions)} actions in {duration_ms:.0f}ms")

            if self.enable_v2:
                self._start_watcher()
                
            # Start Cleanup Task
            self._start_cleanup_task()

        except Exception as e:
            logger.critical(f"Router initialization failed: {e}", exc_info=True)
            raise

    def _start_watcher(self):
        self._watcher = AtomicConfigWatcher(
            config_paths={
                "actions": str(self.loader.action_catalog_path),
                "rules": str(self.loader.rule_config_path),
            },
            on_action_change=self._on_action_config_change,
            on_rule_change=self._on_rule_config_change,
            poll_interval=self.embed_config.reload_poll_interval
        )
        self._watcher.start()

    # --- Background Cleanup Task (M1) ---
    def _start_cleanup_task(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return # No running loop (e.g. in tests or script), handle manually

        self._cleanup_task = loop.create_task(self._cleanup_loop())
        logger.info("Memory cleanup task started")

    async def _cleanup_loop(self):
        while self._running:
            try:
                mem_config = self.loader.get_memory_config()
                interval = mem_config.get("cleanup_interval_seconds", 300)
                
                await asyncio.sleep(interval)
                self.cleanup_expired_contexts()
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(60) # Retry after 1 min on error

    def cleanup_expired_contexts(self):
        """Active cleanup logic with thread safety"""
        now = datetime.datetime.now()
        expired_ids = []
        
        mem_config = self.loader.get_memory_config()
        ttl = mem_config.get("context_ttl_seconds", 1800)

        with self._lock:
            for sid, ctx in self._contexts.items():
                age = (now - ctx.last_updated_at).total_seconds()
                if age > ttl:
                    expired_ids.append(sid)
            
            for sid in expired_ids:
                del self._contexts[sid]
            
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired contexts")

    # --- Event Handlers ---
    def _on_action_config_change(self, path: str):
        start = time.perf_counter()
        logger.info(f"Action config changed: {path}")
        try:
            asyncio.run(self.loader.load())
            self.actions = self.loader.get_all_actions()
            self.embedding_engine.initialize(self.actions) # Re-init for simplicity
            
            duration_ms = (time.perf_counter() - start) * 1000
            self.metrics.record_reload(duration_ms)
        except Exception as e:
            logger.error(f"Error handling config change: {e}", exc_info=True)
            self.metrics.record_error()

    def _on_rule_config_change(self, path: str):
        logger.info(f"Rule config changed: {path}")
        try:
            asyncio.run(self.loader.load())
        except Exception as e:
            logger.error(f"Error reloading rules: {e}", exc_info=True)
            self.metrics.record_error()
    
    def _action_changed(self, old: ActionConfig, new: ActionConfig) -> bool:
        return (old.business_description != new.business_description or 
                old.seed_phrases != new.seed_phrases)

    # --- Context Management ---
    def _get_context(self, session_id: str) -> ConversationContext:
        mem_config = self.loader.get_memory_config()
        ttl = mem_config.get("context_ttl_seconds", 1800)
        
        with self._lock:
            ctx = self._contexts.get(session_id)
            
            if ctx:
                 age = (datetime.datetime.now() - ctx.last_updated_at).total_seconds()
                 if age > ttl:
                     del self._contexts[session_id]
                     ctx = None

            if ctx is None:
                ctx = ConversationContext(session_id=session_id)
                self._contexts[session_id] = ctx
            
            return ctx # Note: Returning ref is okay if caller doesn't mutate heavily without lock, ideally should return copy or handle mutation carefully.
                       # For now, simplistic approach: lock protects the DICT structure (add/remove). 
                       # Mutation of Context object itself is NOT protected here, but in single-threaded async event loop of fastapi it's mostly fine unless threads are involved.
                       # Since we use RLock, if route() runs in threads, we are safer.

    def _check_and_reset_context(self, session_id: str, user_text: str) -> bool:
        text_lower = user_text.lower().strip()
        if any(kw in text_lower for kw in self.RESET_KEYWORDS):
            with self._lock:
                ctx = self._contexts.get(session_id)
                if ctx:
                    ctx.last_action = None
                    ctx.last_domain = None
                    ctx.recent_intents.clear()
                    ctx.last_updated_at = datetime.datetime.now()
                    return True
        return False
    
    def _get_domain_boost(self, ctx: ConversationContext, action_id: str) -> float:
        if not ctx.last_domain: return 0.0
        if not action_id.startswith(ctx.last_domain): return 0.0
        
        weights = self.loader.get_weights()
        return weights.get("domain_boost", 0.1)

    def route(self, request: UserRequest) -> RouterOutput:
        request_id = request.request_id or str(uuid.uuid4())
        session_id = request.session_id or "default_session"
        
        try:
            self._check_and_reset_context(session_id, request.text)
            ctx = self._get_context(session_id)
            
            clean_text = self.preprocessor.process(request.text)
            action_ids = [a.action_id for a in self.actions]
            
            with time_embed():
                embed_scores = self.embedding_engine.batch_score(clean_text, action_ids)
            
            entity_boosts = self.entity_extractor.get_boosts(clean_text, action_ids)
            candidates: List[ActionCandidate] = []
            
            for action in self.actions:
                rule_config = self.loader.get_rule(action.action_id)
                rule_score, rule_reasons = self.rule_engine.score(clean_text, rule_config)
                embed_score = embed_scores.get(action.action_id, 0.0)
                has_pattern = any("pattern" in r for r in rule_reasons)
                
                context_boost = self._get_domain_boost(ctx, action.action_id)
                ent_boost = entity_boosts.get(action.action_id, 0.0)

                intent_type = action.intent_type
                if hasattr(intent_type, 'value'): intent_type = intent_type.value
                else: intent_type = str(intent_type)
                
                final_score = self.fuser.fuse(
                    rule_score=rule_score,
                    embed_score=embed_score,
                    domain=action.domain,
                    intent_type=intent_type,
                    has_pattern=has_pattern,
                    context_boost=context_boost,
                    entity_boost=ent_boost
                )
                
                reasons = rule_reasons.copy()
                if embed_score > 0.5: reasons.append(f"semantic: {embed_score:.2f}")
                if context_boost > 0: reasons.append(f"context: {context_boost:.2f}")
                if ent_boost > 0: reasons.append(f"entity_signal: {ent_boost:.2f}")
                
                candidates.append(ActionCandidate(
                    action_id=action.action_id,
                    friendly_name=action.friendly_name,
                    rule_score=rule_score,
                    embed_score=embed_score,
                    final_score=final_score,
                    reasoning=reasons
                ))
            
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            candidates = self.pairwise_resolver.resolve(clean_text, candidates)
            top_candidates = candidates[:5]
            
            strategy, message = self.ui_decision.decide(top_candidates, self.loader.actions)
            
            # Context Update Logic with Thread Safety
            if top_candidates and top_candidates[0].final_score > 0.6:
                top_action = top_candidates[0].action_id
                # Mutation logic
                # Ideally, Context should be immutable or protected. 
                # Since we are modifying it in-place, and we are in a single request flow:
                # If we assume only one request per session at a time, this is fine.
                # If concurrent requests per session, we need lock on the context object or the map.
                # Here we lock the map to get it, but modification is outside lock.
                # For high concurrency on SAME session, we should lock per session.
                # But for now, global lock for map ops is Step 1.
                ctx.last_action = top_action
                ctx.last_domain = top_action.split(".")[0] if "." in top_action else None
                ctx.recent_intents.append(top_action)
                if len(ctx.recent_intents) > 5: ctx.recent_intents.pop(0)
                ctx.last_updated_at = datetime.datetime.now()

            # Logging
            top_cand = top_candidates[0] if top_candidates else None
            trace = RouterTrace(
                request_id=request_id,
                user_text=request.text,
                semantic_score=top_cand.embed_score if top_cand else 0.0,
                rule_score=top_cand.rule_score if top_cand else 0.0,
                final_score=top_cand.final_score if top_cand else 0.0,
                selected_action=top_cand.action_id if top_cand else "None",
                ui_strategy=strategy,
                timestamp=datetime.datetime.now()
            )
            logger.info(f"ROUTER_TRACE: {trace.model_dump_json()}", extra={"request_id": request_id})

            return RouterOutput(
                request_id=request_id,
                top_actions=top_candidates,
                ui_strategy=strategy,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Route error: {e}", exc_info=True, extra={"request_id": request_id})
            self.metrics.record_error()
            raise ProcessingError(f"Routing failed: {str(e)}")
    
    def reload(self):
        self._initialize()
    
    def shutdown(self):
        self._running = False
        if self._watcher:
            self._watcher.stop()
        if self._cleanup_task:
            self._cleanup_task.cancel()
