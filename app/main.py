import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.models import UserRequest, RouterOutput, ActionState
from app.utils.config_loader import ConfigLoader
from app.router.embed_config import EmbedConfig, VectorStoreType
from app.action_flow.engine import ActionFlowEngine
from app.utils.feedback_logger import log_feedback_event

# Ensure app path is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Hybrid Intent Router v2")

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature flags from environment
# ROUTER_V2_ENABLED removed as V2 is now default

# Config
from app.core.config import settings
from pathlib import Path

# Use settings for paths
loader = ConfigLoader(
    settings.ACTION_CATALOG_PATH,
    settings.KEYWORD_RULES_PATH,
    settings.SYSTEM_CONFIG_PATH
)

# Initialize router based on feature flag
from app.router.router_final import RouterFinal

embed_config = EmbedConfig(
    primary_model=settings.system.embedding_model,
    vector_store=VectorStoreType(settings.system.vector_store),
)
router = RouterFinal(loader, embed_config, enable_v2=True)

@app.on_event("startup")
async def startup_event():
    await router.initialize()

# Initialize Action Flow Engine
afe = ActionFlowEngine(loader)

# Health endpoints
from app.health import router as health_router, set_instances
set_instances(router, getattr(router, 'embedding_engine', None))
app.include_router(health_router)

# --- API Models (Mirrored from main.py) ---

class ActionStartRequest(BaseModel):
    session_id: str
    action_id: str
    initial_text: Optional[str] = ""

class ActionInteractRequest(BaseModel):
    session_id: str
    text: str

class ConfigUpdateRequest(BaseModel):
    content: str

class RouteFeedbackRequest(BaseModel):
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    selected_action: Optional[str] = None
    selection_index: Optional[int] = None
    selection_source: Optional[str] = None
    ui_strategy: Optional[str] = None

class OutcomeFeedbackRequest(BaseModel):
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    action_id: Optional[str] = None
    status: str

# --- Endpoints ---

@app.post("/route", response_model=RouterOutput)
async def route_request(request: UserRequest):
    return router.route(request)

@app.post("/action/start", response_model=ActionState)
def start_action(req: ActionStartRequest):
    try:
        state = afe.start_action(req.session_id, req.action_id, req.initial_text)
        return state
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/action/interact", response_model=ActionState)
def interact_action(req: ActionInteractRequest):
    state = afe.handle_input(req.session_id, req.text)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state

# --- Admin Endpoints ---

@app.get("/admin/config/actions")
def get_action_config():
    return {"content": loader.get_action_catalog_raw()}

@app.post("/admin/config/actions")
def update_action_config(req: ConfigUpdateRequest):
    try:
        loader.update_action_catalog(req.content)
        router.reload()
        return {"status": "updated"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/admin/config/rules")
def get_rule_config():
    return {"content": loader.get_rule_config_raw()}

@app.post("/admin/config/rules")
def update_rule_config(req: ConfigUpdateRequest):
    try:
        loader.update_rule_config(req.content)
        router.reload()
        return {"status": "updated"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Feedback Endpoints ---

@app.post("/feedback/route")
def route_feedback(req: RouteFeedbackRequest):
    log_feedback_event({
        "event_type": "selection",
        "request_id": req.request_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "selected_action": req.selected_action,
        "selection_index": req.selection_index,
        "selection_source": req.selection_source,
        "ui_strategy": req.ui_strategy,
    })
    return {"status": "ok"}

@app.post("/feedback/outcome")
def outcome_feedback(req: OutcomeFeedbackRequest):
    log_feedback_event({
        "event_type": "outcome",
        "request_id": req.request_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "action_id": req.action_id,
        "status": req.status,
    })
    return {"status": "ok"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
