"""FastAPI application for EditMind server and visualizer backend."""

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from editmind.core.registry import EditorRegistry
from editmind.core.types import EditRequest
from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.models.causal_tracer import CausalTracer
from editmind.evaluation.evaluator import KnowledgeEditorEvaluator
from editmind.visualization.causal_plots import generate_svg_causal_heatmap
from editmind.visualization.radar_chart import generate_svg_radar_chart
from editmind.continual import ContinualKnowledgeEditor, generate_svg_interference_matrix
from editmind.safety import KnowledgeConflictDetector, MachineUnlearner
from editmind.server.schemas import EditApiRequest, TraceApiRequest, CompareApiRequest

app = FastAPI(
    title="EditMind API",
    description="Knowledge Editing Framework for Large Language Models",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_GLOBAL_WRAPPER = None
_CONFLICT_DETECTOR = KnowledgeConflictDetector()


def get_model_wrapper() -> UnifiedModelWrapper:
    global _GLOBAL_WRAPPER
    if _GLOBAL_WRAPPER is None:
        _GLOBAL_WRAPPER = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
    return _GLOBAL_WRAPPER


class UnlearnApiRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "The secret password of root is"})
    target_to_erase: str = Field(..., json_schema_extra={"example": "Paris"})
    subject: Optional[str] = Field(None, json_schema_extra={"example": "password"})
    replacement_token: str = Field("unknown", json_schema_extra={"example": "unknown"})
    editor: str = Field("grace", json_schema_extra={"example": "grace"})


class ContinualApiRequest(BaseModel):
    editor: str = Field("grace", json_schema_extra={"example": "grace"})
    requests: List[Dict[str, str]]


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "EditMind Server", "version": "0.2.0"}


@app.get("/api/methods")
def list_methods():
    return {"methods": EditorRegistry.list_available()}


@app.get("/api/models")
def list_models():
    return {
        "models": [
            {"id": "toy_causal_lm", "name": "Toy Transformer LM (6 Layers, 128 Hidden Dim)", "type": "toy"},
            {"id": "gpt2", "name": "GPT-2 Base (124M)", "type": "huggingface"},
            {"id": "gpt2-xl", "name": "GPT-2 XL (1.5B)", "type": "huggingface"},
            {"id": "EleutherAI/gpt-j-6b", "name": "GPT-J (6B)", "type": "huggingface"},
        ]
    }


@app.post("/api/edit")
def perform_edit(req: EditApiRequest):
    wrapper = get_model_wrapper()
    try:
        editor = EditorRegistry.create(req.method, model_wrapper=wrapper)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    edit_req = EditRequest(
        prompt=req.prompt,
        target_new=req.target_new,
        ground_truth=req.ground_truth,
        subject=req.subject,
    )

    result = editor.edit(edit_req)
    _CONFLICT_DETECTOR.register_fact(edit_req)

    return {
        "success": result.success,
        "editor": result.editor_name,
        "execution_time_sec": round(result.execution_time_sec, 4),
        "pre_edit_target_prob": round(result.pre_edit_target_prob, 5),
        "post_edit_target_prob": round(result.post_edit_target_prob, 5),
        "pre_edit_old_prob": round(result.pre_edit_old_prob, 5),
        "post_edit_old_prob": round(result.post_edit_old_prob, 5),
        "delta_norm": round(result.delta_weight_norm, 4),
        "details": result.details,
    }


@app.post("/api/trace")
def perform_causal_trace(req: TraceApiRequest):
    wrapper = get_model_wrapper()
    tracer = CausalTracer(wrapper, noise_std=0.15)
    trace_res = tracer.trace(
        prompt=req.prompt,
        subject=req.subject,
        target=req.target,
    )

    svg_heatmap = generate_svg_causal_heatmap(trace_res)
    best_l, best_t, tok = trace_res.find_critical_layer_and_token()

    return {
        "trace_data": trace_res.to_dict(),
        "svg_heatmap": svg_heatmap,
        "critical_layer": best_l,
        "critical_token": tok,
        "max_indirect_effect": round(float(trace_res.indirect_effects[best_l, best_t]), 4),
    }


@app.post("/api/compare")
def compare_methods_on_edit(req: CompareApiRequest):
    wrapper = get_model_wrapper()
    base_state = wrapper.get_state_dict()

    edit_req = EditRequest(
        prompt=req.prompt,
        target_new=req.target_new,
        ground_truth=req.ground_truth,
        subject=req.subject,
        rephrase_prompts=[f"The true location of {req.subject} is in"],
        locality_prompts=[{"prompt": "The Colosseum is in", "target": "Rome"}],
        portability_prompts=[{"prompt": f"{req.target_new} is where you will find the", "target": req.subject or "it"}],
    )

    comparison_results = {}
    for method in req.methods:
        wrapper.load_state_dict(base_state)
        try:
            editor = EditorRegistry.create(method, model_wrapper=wrapper)
            evaluator = KnowledgeEditorEvaluator(editor)
            metrics = evaluator.evaluate_request(edit_req)
            comparison_results[method] = metrics
        except Exception:
            continue

    wrapper.load_state_dict(base_state)
    svg_radar = generate_svg_radar_chart(comparison_results, size=380)

    return {
        "comparison": {k: v.to_dict() for k, v in comparison_results.items()},
        "svg_radar": svg_radar,
    }


@app.post("/api/unlearn")
def perform_unlearning(req: UnlearnApiRequest):
    wrapper = get_model_wrapper()
    unlearner = MachineUnlearner(wrapper, editor_name=req.editor)
    res = unlearner.unlearn_fact(
        prompt=req.prompt,
        target_to_erase=req.target_to_erase,
        subject=req.subject,
        replacement_token=req.replacement_token,
    )
    return res.to_dict()


@app.post("/api/continual")
def perform_continual_stream(req: ContinualApiRequest):
    wrapper = get_model_wrapper()
    base_state = wrapper.get_state_dict()

    editor = EditorRegistry.create(req.editor, model_wrapper=wrapper)
    continual = ContinualKnowledgeEditor(editor)

    edit_requests = [
        EditRequest(
            prompt=item["prompt"],
            target_new=item["target_new"],
            ground_truth=item.get("ground_truth"),
            subject=item.get("subject"),
        )
        for item in req.requests
    ]

    trajectory = continual.run_sequential_stream(edit_requests)
    svg_matrix = generate_svg_interference_matrix(trajectory)
    wrapper.load_state_dict(base_state)

    return {
        "trajectory": trajectory.to_dict(),
        "svg_matrix": svg_matrix,
    }


@app.post("/api/check-conflict")
def check_knowledge_conflict(req: EditApiRequest):
    edit_req = EditRequest(
        prompt=req.prompt,
        target_new=req.target_new,
        ground_truth=req.ground_truth,
        subject=req.subject,
    )
    report = _CONFLICT_DETECTOR.check_request(edit_req)
    return report.to_dict()


dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/static", StaticFiles(directory=dashboard_path), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_file = os.path.join(dashboard_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return "<h1>EditMind Dashboard</h1>"
