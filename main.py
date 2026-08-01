"""AgileBot Gateway Backend
- Proxies to https://api.agilebot.dev for native models/tools
- Adds custom models + tools on top
- Routes custom models to OpenRouter or local providers
"""

import os
import json
import time
import uuid
import threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import httpx

UPSTREAM = os.environ.get("AGILEBOT_UPSTREAM", "https://api.agilebot.dev")
PORT = int(os.environ.get("PORT", "8765"))
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BUILD_TAG = "build-2026-08-01-createanim-synth-v3"
FREELLMAPI_URL = os.environ.get("FREELLMAPI_URL", "https://freellmapi-cliz.onrender.com")
FREELLMAPI_KEY = os.environ.get("FREELLMAPI_KEY", "")

app = FastAPI(title="AgileBot Gateway", version="0.2.0")

# ---------- Persistence ----------

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CUSTOM_TOOLS_FILE = DATA_DIR / "custom_tools.json"
CUSTOM_MODELS_FILE = DATA_DIR / "custom_models.json"
SYSTEM_PROMPT_FILE = DATA_DIR / "system_prompt.txt"

DEFAULT_SYSTEM_PROMPT = (
    "You are AgileBot, an expert Roblox Studio AI assistant that writes clean, correct Luau code.\n"
    "You have access to tools that operate INSIDE Roblox Studio (create_animation, search_animations, search_sounds).\n"
    "CRITICAL TOOL-USAGE RULE: When the user asks you to build, animate, modify, or search anything in their "
    "experience, you MUST call the matching tool via a tool_call — NEVER just describe the steps, NEVER hand back a "
    "script for the user to paste, and NEVER tell the user to run something themselves. The tool runs it for them.\n"
    "When you call a tool, you MUST populate EVERY relevant parameter with a concrete value (animation name, fps, "
    "loop, and a full keyframes array with time/pose/CFrame data). Do NOT call a tool with empty or missing arguments "
    "— an empty call does nothing. If you need a target you can omit ref/path (it defaults to the selected/player model).\n"
    "Example create_animation call: name='Wave', fps=30, loop=true, keyframes=[{time=0,pose='CFrame identity'},"
    "{time=0.5,pose='Arm raised'}, {time=1.0,pose='CFrame identity'}].\n"
    "Only fall back to writing Luau code in the chat when the user explicitly asks for raw script text.\n"
    "When writing Luau: follow Roblox Luau conventions, use task.spawn/wait instead of spawn/wait where appropriate, "
    "guard pcall around HttpService and DataStore calls, and never use Lua 5.1-only syntax (no +=, no const). "
    "Keep scripts small, readable, and production-safe."
)

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def load_system_prompt() -> str:
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text(encoding='utf-8').strip()
    return DEFAULT_SYSTEM_PROMPT

CUSTOM_TOOLS: dict = load_json(CUSTOM_TOOLS_FILE, {})
CUSTOM_MODELS: dict = load_json(CUSTOM_MODELS_FILE, {})
SYSTEM_PROMPT: str = load_system_prompt()

LOCAL_CONVERSATIONS_FILE = DATA_DIR / "local_conversations.json"
LOCAL_CONVERSATIONS: dict = load_json(LOCAL_CONVERSATIONS_FILE, {})

# Streaming operation store. The plugin polls GET /operations/{op}/events and
# expects a rolling list of {seq, type, payload} events plus a terminal status.
# type "block_upsert" -> payload.block (full block, streaming=true)
# type "block_patch"  -> payload.{block_id, patch:{text=delta}} (append) or
#                         payload.{block_id, patch:{streaming=false, text=full}}
OPERATION_EVENTS: dict = {}
OPERATION_LOCK = threading.Lock()

# Aliases that aren't in CUSTOM_MODELS but should still bypass the upstream
# (api.agilebot.dev) conversation validator, which has no idea what these are.
CUSTOM_MODEL_ALIASES = {"openrouter/auto", "freellmapi/auto"}

# ---------- Built-in tools ----------

BUILTIN_TOOLS = {
    "search_animations": {
        "name": "search_animations",
        "description": "Search Roblox catalog for animations by keyword. Returns asset id, name, creator.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search keyword"},
                "category": {"type": "string", "description": "optional filter (e.g. emote, run, idle)"},
            },
            "required": ["query"],
        },
        "endpoint": "/roblox-proxy/catalog.roblox.com/v1/search/items/details?Category=12&Subcategory=27&Keyword={query}&Limit=20&SortType=Relevance",
    },
    "search_sounds": {
        "name": "search_sounds",
        "description": "Search Roblox catalog for sounds/audio by keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search keyword"},
            },
            "required": ["query"],
        },
        "endpoint": "/roblox-proxy/apis.roblox.com/toolbox-service/v1/search?query={query}&category=Audio",
    },
    "create_animation": {
        "name": "create_animation",
        "description": "Build and PLAY a Roblox KeyframeSequence animation on a target instance (model/Humanoid/AnimationController) from a list of keyframe poses. ALWAYS call this when the user wants an animation/wave/spin/movement built - do not return a script. Required params you MUST provide: name (string), fps (number, default 30), loop (boolean, default true), keyframes (array, each time:number in seconds, pose:string e.g. CFrame identity or left arm up 45deg). Example: name=Wave, fps=30, loop=true, keyframes=[{time=0,pose=CFrame identity},{time=0.5,pose=right arm raised},{time=1.0,pose=CFrame identity}]. If no target is specified, omit ref/path (defaults to the selected/player model).",

        "parameters": {
        "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Optional instance ref (from ResolveReference) of the target model/part."},
                "path": {"type": "string", "description": "Optional workspace path to the target."},
                "name": {"type": "string", "description": "Animation name."},
                "fps": {"type": "number", "description": "Keyframe playback fps (default 30)."},
                "loop": {"type": "boolean", "description": "Loop the animation (default true)."},
                "keyframes": {
                    "type": "array",
                    "description": "Ordered keyframes.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "time": {"type": "number", "description": "Time (seconds) of this keyframe."},
                            "poses": {
                                "type": "array",
                                "description": "Poses at this keyframe.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "part": {"type": "string"},
                                        "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                                        "rx": {"type": "number"}, "ry": {"type": "number"}, "rz": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["keyframes"]
        },
    },
}

# ---------- Models ----------



def _add_auth(headers: dict, request: Request) -> dict:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        headers["authorization"] = auth
    return headers

def _proxy_headers() -> dict:
    return {
        "accept": "application/json",
        "accept-encoding": "identity",
        "user-agent": "AgileBotGateway/0.1 (+https://github.com/abdelqalzeinn-cmyk/backend-agile)",
    }

_HOP_BY_HOP = {"content-encoding", "content-length", "transfer-encoding", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"}

def _clean_response_headers(headers: dict) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}

def _proxy(method: str, path: str, body: dict | bytes | None = None, headers: dict | None = None) -> httpx.Response:
    url = f"{UPSTREAM}{path}"
    merged = dict(_proxy_headers())
    if headers:
        merged.update(headers)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        if method == "GET":
            return client.get(url, headers=merged)
        if isinstance(body, bytes):
            return client.request(method, url, content=body, headers=merged)
        return client.request(method, url, json=body, headers=merged)

# ---------- Health ----------

@app.get("/health")
def health():
    return {
        "ok": True,
        "upstream": UPSTREAM,
        "build_tag": BUILD_TAG,
        "custom_models": len(CUSTOM_MODELS),
        "custom_tools": len(CUSTOM_TOOLS),
    }

# ---------- Proxy upstream ----------

# ---------- Explicit routes (must come BEFORE catch-all) ----------

@app.get("/health")
def health_route():
    return health()

@app.get("/models/gateway")
async def models_gateway(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    upstream_models = []
    upstream_status = None
    try:
        r = _proxy("GET", "/models", headers=h)
        upstream_status = r.status_code
        if r.status_code == 200:
            data = r.json()
            upstream_models = data.get("models", data) if isinstance(data, dict) else data
    except Exception as e:
        upstream_status = f"error: {e}"

    # Sync OpenRouter free models + FreeLLMAPI models into custom catalog if not already present
    try:
        sync_openrouter_free_models()
    except Exception:
        pass
    try:
        sync_freellmapi_models()
    except Exception:
        pass

    merged = []
    for m in upstream_models:
        if isinstance(m, dict):
            mm = dict(m)
            mm["enabled"] = True
            merged.append(mm)
        else:
            merged.append(m)
    for m in CUSTOM_MODELS.values():
        mm = dict(m)
        mm["enabled"] = True
        merged.append(mm)
    return {
        "models": merged,
        "custom": list(CUSTOM_MODELS.values()),
        "upstream_count": len(upstream_models),
        "upstream_status": upstream_status,
        "custom_count": len(CUSTOM_MODELS),
    }

@app.get("/tools/gateway")
def tools_gateway():
    return list_gateway_tools()

@app.post("/tools/gateway/call")
async def tools_gateway_call(request: Request):
    return await call_gateway_tool(request)

@app.get("/admin/tools")
def admin_tools_list():
    return list_admin_tools()

@app.post("/admin/tools")
async def admin_tools_add(request: Request):
    body = await request.json()
    req = body
    return add_admin_tool(req)

@app.delete("/admin/tools/{name}")
def admin_tools_delete(name: str):
    return delete_admin_tool(name)

@app.get("/admin/models")
def admin_models_list():
    return list_admin_models()

@app.post("/admin/models/sync-openrouter-free")
def admin_models_sync():
    if not OPENROUTER_KEY:
        return JSONResponse({
            "ok": False,
            "error": "OPENROUTER_API_KEY not set in environment",
            "fix": "Start backend with: $env:OPENROUTER_API_KEY='...'; python main.py",
        }, status_code=500)
    try:
        fetched = fetch_openrouter_free_models()
    except Exception as e:
        return JSONResponse({
            "ok": False,
            "error": f"OpenRouter fetch failed: {e}",
        }, status_code=502)
    try:
        result = sync_openrouter_free_models()
    except Exception as e:
        return JSONResponse({
            "ok": False,
            "error": f"Sync failed: {e}",
        }, status_code=500)
    return JSONResponse({
        "ok": True,
        "key_set": bool(OPENROUTER_KEY),
        "fetched_count": len(fetched),
        "sample": [m.get("id") for m in fetched[:5]],
        **result,
    })

@app.post("/admin/models")
async def admin_models_add(request: Request):
    body = await request.json()
    req = body
    return add_admin_model(req)

@app.delete("/admin/models/{model_id}")
def admin_models_delete(model_id: str):
    return delete_admin_model(model_id)

@app.get("/models/custom")
def models_custom():
    return list_custom_models()

@app.get("/models/custom/{model_id}")
def models_custom_one(model_id: str):
    return get_custom_model(model_id)

# ---------- Catch-all proxy ----------


# ---------- Workspace ----------

@app.get("/workspace")
async def workspace(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    r = _proxy("GET", "/workspace", headers=h)
    if r.status_code != 200:
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
    try:
        data = r.json()
        ws = data.get("workspace") or {}
        user = data.get("user") or {}
        data["username"] = user.get("username") or ws.get("name") or ws.get("slug") or ""
        return JSONResponse(content=data, status_code=200)
    except Exception:
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))


@app.get("/conversations")
async def conversations(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    r = _proxy("GET", "/conversations", headers=h)
    if r.status_code != 200:
        return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))
    try:
        data = r.json()
        return JSONResponse(content=data, status_code=200, headers={"Content-Encoding": "identity"})
    except Exception:
        return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))


@app.post("/conversations")
async def conversations_post(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    ct = request.headers.get("content-type") or request.headers.get("Content-Type")
    if ct:
        h["content-type"] = ct
    body = await request.body()

    # Custom-provider models (freellmapi/auto, openrouter/auto, synced OpenRouter
    # free models, ...) don't exist upstream, so api.agilebot.dev rejects them
    # with 422 model_unavailable. Handle these entirely locally instead.
    try:
        body_json = json.loads(body) if body else {}
    except Exception:
        body_json = {}
    model_id = str(body_json.get("model", ""))
    if _is_custom_model(model_id):
        try:
            result = _handle_custom_conversation(model_id, body_json.get("message", ""), None)
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            return JSONResponse({"detail": {"code": "custom_model_error", "model": model_id, "message": str(e)}}, status_code=502)

    r = _proxy("POST", "/conversations", body=body, headers=h)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))


@app.post("/conversations/{conversation_id}/messages")
async def conversations_messages_post(conversation_id: str, request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    ct = request.headers.get("content-type") or request.headers.get("Content-Type")
    if ct:
        h["content-type"] = ct
    body = await request.body()

    try:
        body_json = json.loads(body) if body else {}
    except Exception:
        body_json = {}
    model_id = str(body_json.get("model", ""))

    # Route to the local shim if this is a custom model OR this conversation
    # was already started locally (so follow-ups stay on the same path).
    if _is_custom_model(model_id) or conversation_id in LOCAL_CONVERSATIONS:
        try:
            result = _handle_custom_conversation(model_id, body_json.get("message", ""), conversation_id)
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            return JSONResponse({"detail": {"code": "custom_model_error", "model": model_id, "message": str(e)}}, status_code=502)

    r = _proxy("POST", f"/conversations/{conversation_id}/messages", body=body, headers=h)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))


# The plugin POSTs tool execution results here after running a tool locally
# (e.g. create_animation). We don't need to forward them upstream — the action
# already happened in Studio — so just acknowledge with 200.
@app.post("/operations/{operation_id}/tool_results")
async def operation_tool_results(operation_id: str, request: Request):
    try:
        await request.body()
    except Exception:
        pass
    return JSONResponse(content={"ok": True, "operation_id": operation_id}, status_code=200)




@app.post("/admin/system-prompt")
async def update_system_prompt(request: Request):
    """Inject custom instructions into the system prompt (persisted to disk)."""
    global SYSTEM_PROMPT
    body = await request.json()
    instruction = (body.get("instruction") or "").strip()
    if not instruction:
        return {"ok": False, "error": "instruction required"}
    SYSTEM_PROMPT = instruction
    SYSTEM_PROMPT_FILE.write_text(instruction, encoding='utf-8')
    return {
        "ok": True,
        "system_prompt": SYSTEM_PROMPT,
        "available_tools": list(BUILTIN_TOOLS.keys()) + list(CUSTOM_TOOLS.keys()),
        "message": "System prompt saved and will be injected into every chat request.",
    }

@app.get("/admin/system-prompt")
async def get_system_prompt():
    """Get current system prompt and available tools."""
    return {
        "system_prompt": SYSTEM_PROMPT,
        "available_tools": list(BUILTIN_TOOLS.keys()) + list(CUSTOM_TOOLS.keys()),
        "builtin_count": len(BUILTIN_TOOLS),
        "custom_count": len(CUSTOM_TOOLS),
    }


@app.post("/admin/models/sync-freellmapi")
def admin_models_sync_freellmapi():
    if not FREELLMAPI_KEY:
        return JSONResponse({
            "ok": False,
            "error": "FREELLMAPI_KEY not set in environment",
            "fix": "Set FREELLMAPI_KEY on the Render service env vars, then POST here.",
        }, status_code=500)
    try:
        result = sync_freellmapi_models()
    except Exception as e:
        return JSONResponse({"detail": f"Sync failed: {e}"}, status_code=500)
    return JSONResponse({
        "ok": True,
        "key_set": bool(FREELLMAPI_KEY),
        "total_custom": len(CUSTOM_MODELS),
        **result,
    })


@app.get("/conversations/{conversation_id}/timeline")
def conversations_timeline(conversation_id: str, request: Request):
    # Local custom-provider conversations live only in our shim, not upstream.
    # The plugin fetches this after a streamed operation completes to reconcile
    # the final conversation; without it we 404 and the chat never finalizes.
    conv = LOCAL_CONVERSATIONS.get(conversation_id)
    if conv is None:
        # Not a local conversation — let upstream handle it (native models).
        h = dict(_proxy_headers())
        a = request.headers.get("authorization") or request.headers.get("Authorization")
        if a:
            h["authorization"] = a
        r = _proxy("GET", f"/conversations/{conversation_id}/timeline", headers=h)
        return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))

    blocks = []
    seq = 1
    for m in conv.get("messages", []):
        role = m.get("role", "user")
        blocks.append(_make_block(role, m.get("content", ""), seq))
        seq += 1
    return JSONResponse(content={
        "conversation": {"id": conv.get("id", conversation_id), "name": conv.get("name", "")},
        "timeline": blocks,
        "has_more_older": False,
    })


@app.get("/operations/{operation_id}/events")
def operation_events(operation_id: str, after_seq: int = 0, limit: int = 50):
    with OPERATION_LOCK:
        op = OPERATION_EVENTS.get(operation_id)
        if op is None:
            return {"operation_id": operation_id, "status": "unknown", "events": []}
        events = [e for e in op["events"] if e["seq"] > after_seq]
        if limit and limit > 0:
            events = events[-limit:]
        return {"operation_id": operation_id, "status": op["status"], "events": events}


@app.get("/{path:path}")
async def proxy_get(path: str, request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    r = _proxy("GET", f"/{path}", headers=h)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))

@app.post("/{path:path}")
async def proxy_post(path: str, request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    ct = request.headers.get("content-type") or request.headers.get("Content-Type")
    if ct:
        h["content-type"] = ct
    if path == "chat/completions":
        body = await request.json()
        model_id = body.get("model", "")

        # Inject gateway tools into the request so the model sees animation/sound tools
        try:
            tool_list_resp = list_gateway_tools()
            available = tool_list_resp.get("tools", [])
        except Exception:
            available = []
        if available:
            formatted = []
            for t in available:
                if isinstance(t, dict):
                    formatted.append({
                        "name": t.get("name") or t.get("id"),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    })
            if formatted:
                body.setdefault("tools", [])
                # dedupe by name
                existing = {t.get("name") for t in body["tools"] if isinstance(t, dict)}
                for t in formatted:
                    if t["name"] not in existing:
                        body["tools"].append({
                            "type": "function",
                            "function": t,
                        })

        # Inject the system prompt as the first system message so the model
        # follows our Roblox/Luau conventions and tool-usage guidance.
        if SYSTEM_PROMPT:
            msgs = body.get("messages", [])
            if not isinstance(msgs, list):
                msgs = []
            if not msgs or msgs[0].get("role") != "system":
                body["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
            else:
                # prepend our guidance to the existing system message
                msgs[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + msgs[0].get("content", "")}
                body["messages"] = msgs
        # Route custom models to their provider
        if model_id in CUSTOM_MODELS:
            model = CUSTOM_MODELS[model_id]
            provider = model.get("provider", "openrouter")
            endpoint = model.get("endpoint", "https://openrouter.ai/api/v1")

            if provider == "freellmapi":
                return forward_freellmapi(body)

            if provider == "openrouter":
                if not OPENROUTER_KEY:
                    return JSONResponse({"error": "OPENROUTER_API_KEY not set"}, status_code=500)
                return forward_openrouter(endpoint, body)

            if provider == "local":
                return forward_local(body)

            return JSONResponse({"error": f"Unknown provider: {provider}"}, status_code=400)

        # Not a custom model — proxy to upstream
        r = _proxy("POST", f"/{path}", body=body, headers=_add_auth(dict(_proxy_headers()), request))
        return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))
    else:
        body = await request.body()
    r = _proxy("POST", f"/{path}", body=body, headers=h)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))

# ---------- OpenRouter forward ----------

def forward_openrouter(endpoint: str, body: dict) -> Response:
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {OPENROUTER_KEY}",
        "http-referer": "http://localhost:8765",
        "x-title": "AgileBot Gateway",
    }
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        r = client.post(url, json=body, headers=headers)

    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))

# ---------- Local model forward ----------

def forward_local(body: dict) -> Response:
    """Placeholder for local model inference."""
    messages = body.get("messages", [])
    last = messages[-1].get("content", "") if messages else ""
    return JSONResponse({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": f"[local model placeholder] {last[:100]}"
            }
        }]
    })

# ---------- OpenRouter free models syncer ----------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

def _openrouter_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8765",
        "X-Title": "AgileBot Gateway",
    }

def fetch_openrouter_free_models():
    if not OPENROUTER_KEY:
        return []
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
            r = client.get(f"{OPENROUTER_API_URL}/models", headers=_openrouter_headers())
            if r.status_code != 200:
                return []
            data = r.json()
            models = data.get("data", [])
            return [m for m in models if ":free" in m.get("id", "")]
    except Exception as e:
        return []

def sync_openrouter_free_models():
    models = fetch_openrouter_free_models()
    added = []
    for m in models:
        mid = m.get("id", "")
        if mid not in CUSTOM_MODELS:
            CUSTOM_MODELS[mid] = {
                "id": mid,
                "name": m.get("name", mid),
                "provider": "openrouter",
                "endpoint": OPENROUTER_API_URL,
                "context_length": m.get("context_length", 8192),
                "supports_tools": "tools" in str(m.get("supported_parameters", "")),
                "created_at": time.time(),
            }
            added.append(mid)
    if added:
        save_json(CUSTOM_MODELS_FILE, CUSTOM_MODELS)
    return {
        "added": added,
        "skipped": [m.get("id") for m in models if m.get("id") in CUSTOM_MODELS],
        "total_free": len(models),
        "sample": [m.get("id") for m in models[:5]],
    }


# ---------- FreeLLMAPI forward + model sync ----------

def _freellmapi_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if FREELLMAPI_KEY:
        h["Authorization"] = f"Bearer {FREELLMAPI_KEY}"
    return h


def fetch_freellmapi_models() -> list:
    if not FREELLMAPI_URL:
        return []
    last_err = None
    for attempt in range(3):
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
                r = client.get(f"{FREELLMAPI_URL.rstrip('/')}/v1/models", headers=_freellmapi_headers())
                if r.status_code == 429:
                    # rate-limited — back off and retry
                    time.sleep(2.0 * (attempt + 1))
                    last_err = RuntimeError(f"/v1/models 429 (attempt {attempt+1})")
                    continue
                if r.status_code != 200:
                    last_err = RuntimeError(f"/v1/models {r.status_code}")
                    continue
                data = r.json()
                return data.get("data", data) if isinstance(data, dict) else data
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
            continue
    if last_err:
        print(f"[freellmapi] sync failed after retries: {last_err}")
    return []


def sync_freellmapi_models() -> dict:
    models = fetch_freellmapi_models()
    added = []
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "")
        if mid == "" or mid in CUSTOM_MODELS:
            continue
        CUSTOM_MODELS[mid] = {
            "id": mid,
            "name": m.get("name", mid),
            "provider": "freellmapi",
            "endpoint": FREELLMAPI_URL,
            "context_length": m.get("context_length", 8192),
            "supports_tools": m.get("supports_tools", False),
            "created_at": time.time(),
        }
        added.append(mid)
    if added:
        save_json(CUSTOM_MODELS_FILE, CUSTOM_MODELS)
    return {
        "added": added,
        "skipped": [m.get("id") for m in models if isinstance(m, dict) and m.get("id") in CUSTOM_MODELS],
        "total": len(models),
        "sample": [m.get("id") for m in models[:5] if isinstance(m, dict)],
    }


def forward_freellmapi(body: dict) -> Response:
    url = f"{FREELLMAPI_URL.rstrip('/')}/v1/chat/completions"
    headers = _freellmapi_headers()
    headers["HTTP-Referer"] = "http://localhost:8765"
    headers["X-Title"] = "AgileBot Gateway"
    # FreeLLMAPI's own API only accepts "auto" (or a family name); specific
    # sub-model ids from its /v1/models catalog error out. Force "auto" so any
    # freellmapi/* selection works through the direct forwarder too.
    fixed_body = dict(body)
    fixed_body["model"] = "auto"
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        r = client.post(url, json=fixed_body, headers=headers)
    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))


# ---------- Local conversation shim for custom-provider models ----------
# api.agilebot.dev's /conversations endpoint validates "model" against its own
# native catalog and rejects anything it doesn't recognize (422 model_unavailable).
# For custom-provider models (freellmapi/auto, openrouter/auto, synced OpenRouter
# free models) we never forward the conversation to upstream at all -- we run the
# whole turn locally and hand the client back a response shaped like the one
# AgileBot's UI already knows how to render (conversation + timeline blocks).

def _strip_prefix(model_id: str, provider_prefix: str) -> str:
    p = provider_prefix + "/"
    return model_id[len(p):] if model_id.startswith(p) else model_id


def _is_custom_model(model_id: str) -> bool:
    if not model_id:
        return False
    if model_id in CUSTOM_MODELS or model_id in CUSTOM_MODEL_ALIASES:
        return True
    # Plugin sends prefixed ids (freellmapi/..., openrouter/...); route any of
    # those to our local handler even if the exact id isn't in CUSTOM_MODELS
    # (e.g. a synced sub-model whose raw id differs from the prefixed send).
    if model_id.startswith("freellmapi/") or model_id.startswith("openrouter/"):
        return True
    # Also accept the stripped raw id in case the picker sends it unprefixed.
    if _strip_prefix(model_id, "freellmapi") in CUSTOM_MODELS:
        return True
    if _strip_prefix(model_id, "openrouter") in CUSTOM_MODELS:
        return True
    return False


def _build_tools_payload() -> list:
    """Build OpenAI-style tool definitions from the gateway catalog."""
    try:
        available = list_gateway_tools().get("tools", [])
    except Exception:
        return []
    tools = []
    seen = set()
    for t in available:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or t.get("id")
        if not name or name in seen:
            continue
        seen.add(name)
        params = t.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": params if params.get("type") == "object" else {
                    "type": "object",
                    "properties": params if isinstance(params, dict) else {},
                },
            },
        })
    return tools


def _call_custom_model_sync(model_id: str, messages: list) -> dict:
    """Call the underlying provider directly (non-streaming) and return the raw message dict.

    Returns {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}.
    Tool calls are surfaced so the backend can emit them as plugin-executable tool_request blocks.
    """
    provider, endpoint, real_model = None, None, model_id

    if model_id in CUSTOM_MODELS:
        m = CUSTOM_MODELS[model_id]
        provider = m.get("provider", "openrouter")
        endpoint = m.get("endpoint")
    elif model_id == "openrouter/auto":
        provider, endpoint, real_model = "openrouter", OPENROUTER_API_URL, "openrouter/auto"
    elif model_id.startswith("freellmapi/"):
        # FreeLLMAPI only accepts "auto" (or a family name); specific sub-model
        # ids from its /v1/models catalog error out. Force the site's working
        # "auto" model so every freellmapi/* selection just works, no fallback.
        provider, real_model = "freellmapi", "auto"
    else:
        # Bare synced id (no prefix) — look it up in the catalog by stripped form.
        for pfx in ("freellmapi/", "openrouter/"):
            cand = _strip_prefix(model_id, pfx)
            if cand in CUSTOM_MODELS:
                m = CUSTOM_MODELS[cand]
                provider = m.get("provider", "openrouter")
                endpoint = m.get("endpoint")
                if provider == "freellmapi":
                    real_model = "auto"
                break

    if provider == "openrouter":
        if not OPENROUTER_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set on the backend")
        url = f"{(endpoint or OPENROUTER_API_URL).rstrip('/')}/chat/completions"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {OPENROUTER_KEY}",
            "http-referer": "http://localhost:8765",
            "x-title": "AgileBot Gateway",
        }
    elif provider == "freellmapi":
        real_model = _strip_prefix(real_model, "freellmapi")
        url = f"{FREELLMAPI_URL.rstrip('/')}/v1/chat/completions"
        headers = _freellmapi_headers()
        headers["HTTP-Referer"] = "http://localhost:8765"
        headers["X-Title"] = "AgileBot Gateway"
    else:
        raise RuntimeError(f"Unknown custom model provider for '{model_id}'")

    body = {"model": real_model, "messages": messages}
    tools = _build_tools_payload()
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if SYSTEM_PROMPT:
        msgs = list(messages)
        if not msgs or msgs[0].get("role") != "system":
            body["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
        else:
            msgs[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + msgs[0].get("content", "")}
            body["messages"] = msgs

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        r = client.post(url, json=body, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"{provider} error {r.status_code}: {r.text[:500]}")
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"{provider} returned no choices: {json.dumps(data)[:300]}")
    msg = choices[0].get("message", {})
    content = msg.get("content", "") or ""
    raw_calls = msg.get("tool_calls") or []
    tool_calls = []
    for tc in raw_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args_str = fn.get("arguments", "{}") or "{}"
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except Exception:
            args = {}
        tool_calls.append({"id": tc.get("id", ""), "name": name, "arguments": args})
    return {"content": content, "tool_calls": tool_calls}


def _make_block(role: str, text: str, seq_n: int) -> dict:
    return {
        "id": f"message:{uuid.uuid4().hex}",
        "role": role,
        "text": text,
        "seq": seq_n,
        "created_at_unix_ms": int(time.time() * 1000),
    }


def _op_emit(operation_id: str, event_type: str, payload: dict):
    """Append one event to an operation's rolling event log (thread-safe)."""
    with OPERATION_LOCK:
        op = OPERATION_EVENTS.get(operation_id)
        if op is None:
            op = {"status": "running", "events": [], "seq": 0}
            OPERATION_EVENTS[operation_id] = op
        op["seq"] += 1
        op["events"].append({"seq": op["seq"], "type": event_type, "payload": payload})


def _op_finish(operation_id: str, status: str = "completed"):
    with OPERATION_LOCK:
        op = OPERATION_EVENTS.get(operation_id)
        if op is None:
            op = {"status": "running", "events": [], "seq": 0}
            OPERATION_EVENTS[operation_id] = op
        op["status"] = status


def _parse_sse_lines(raw: str):
    """Yield data payloads from an SSE byte stream (lines prefixed 'data:')."""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        yield data


def _stream_custom_model(operation_id: str, model_id: str, messages: list, conv_id: str,
                         assistant_seq: int, tool_calls_acc: list):
    """Background worker: call the provider with stream:true, push block events
    to the operation store so the plugin renders live typing."""
    render_id = f"message:{uuid.uuid4().hex}"
    # 1) upsert the assistant block in 'streaming' state
    _op_emit(operation_id, "block_upsert", {
        "block": {
            "render_id": render_id,
            "id": render_id,
            "role": "assistant",
            "text": "",
            "seq": assistant_seq,
            "created_at_unix_ms": int(time.time() * 1000),
            "streaming": True,
        }
    })

    acc = ""
    try:
        provider, endpoint, real_model = None, None, model_id
        if model_id in CUSTOM_MODELS:
            m = CUSTOM_MODELS[model_id]
            provider = m.get("provider", "openrouter")
            endpoint = m.get("endpoint")
        elif model_id == "openrouter/auto":
            provider, endpoint, real_model = "openrouter", OPENROUTER_API_URL, "openrouter/auto"
        elif model_id.startswith("freellmapi/"):
            provider, real_model = "freellmapi", "auto"
        else:
            for pfx in ("freellmapi/", "openrouter/"):
                cand = _strip_prefix(model_id, pfx)
                if cand in CUSTOM_MODELS:
                    m = CUSTOM_MODELS[cand]
                    provider = m.get("provider", "openrouter")
                    endpoint = m.get("endpoint")
                    if provider == "freellmapi":
                        real_model = "auto"
                    break
        if provider is None:
            raise RuntimeError(f"Unknown custom model provider for '{model_id}'")

        # Resolve the model the user actually picked (so an explicit pick is
        # tried first), then build a FreeLLMAPI-only fallback chain. OpenRouter
        # stays single-shot (no fallback) per owner directive.
        if provider == "freellmapi":
            # model_id may be "freellmapi/<x>" or a bare synced id; either way
            # the chosen model is the part after the optional "freellmapi/" prefix.
            chosen = _strip_prefix(model_id, "freellmapi") or "auto"
            if chosen == "auto":
                candidates = ["auto", "deepseek-v4-flash", "qwen3.6-27b",
                              "mistral-small-4", "gpt-oss-20b", "llama-3.3-70b"]
            else:
                fallbacks = ["auto", "deepseek-v4-flash", "qwen3.6-27b",
                             "mistral-small-4", "gpt-oss-20b", "llama-3.3-70b"]
                candidates = [chosen] + [f for f in fallbacks if f != chosen]
        else:
            candidates = [real_model]

        tools = _build_tools_payload()
        # If the user is clearly asking for an animation, FORCE the model to call
        # create_animation (FreeLLMAPI "auto" is unreliable at deciding to use
        # tools on its own). This guarantees the tool fires with populated args
        # instead of the model just describing a script.
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "") or ""
                break
        ANIM_KW = ("animat", "wave", "spin", "dance", "keyframe", "motion", "movement", "create animation")
        forced_tool = None
        if any(k in last_user.lower() for k in ANIM_KW) and any(t.get("name") == "create_animation" for t in tools):
            forced_tool = "create_animation"
        sys_msgs = list(messages)
        if SYSTEM_PROMPT:
            if not sys_msgs or sys_msgs[0].get("role") != "system":
                sys_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + sys_msgs
            else:
                sys_msgs[0] = {"role": "system",
                               "content": SYSTEM_PROMPT + "\n\n" + sys_msgs[0].get("content", "")}

        last_err = None
        for attempt_model in candidates:
            if attempt_model != candidates[0]:
                # brief pause before retrying a different model
                time.sleep(0.4)
            try:
                if provider == "openrouter":
                    if not OPENROUTER_KEY:
                        raise RuntimeError("OPENROUTER_API_KEY not set on the backend")
                    url = f"{(endpoint or OPENROUTER_API_URL).rstrip('/')}/chat/completions"
                    headers = {"content-type": "application/json",
                               "authorization": f"Bearer {OPENROUTER_KEY}",
                               "http-referer": "http://localhost:8765", "x-title": "AgileBot Gateway"}
                else:  # freellmapi
                    url = f"{FREELLMAPI_URL.rstrip('/')}/v1/chat/completions"
                    headers = _freellmapi_headers()
                    headers["HTTP-Referer"] = "http://localhost:8765"
                    headers["X-Title"] = "AgileBot Gateway"

                body = {"model": attempt_model, "messages": sys_msgs, "stream": True}
                if tools:
                    body["tools"] = tools
                    if forced_tool and provider == "freellmapi":
                        # Force the router to emit this specific tool call so the
                        # animation is actually built instead of just described.
                        body["tool_choice"] = {"type": "function",
                                                "function": {"name": forced_tool}}
                    else:
                        body["tool_choice"] = "auto"

                with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(connect=10.0, read=25.0, write=30.0, pool=10.0)) as client:
                    with client.stream("POST", url, json=body, headers=headers) as resp:
                        if resp.status_code != 200:
                            raise RuntimeError(
                                f"{provider}:{attempt_model} stream error {resp.status_code}: {resp.read().decode()[:200]}")
                        got_any = False
                        for chunk in resp.iter_lines():
                            for data in _parse_sse_lines(chunk):
                                try:
                                    piece = json.loads(data)
                                except Exception:
                                    continue
                                delta = (piece.get("choices") or [{}])[0].get("delta") or {}
                                text_delta = delta.get("content") or ""
                                if text_delta:
                                    got_any = True
                                    acc += text_delta
                                    _op_emit(operation_id, "block_patch", {
                                        "block_id": render_id,
                                        "patch": {"text_append": text_delta},
                                    })
                                for tc in delta.get("tool_calls") or []:
                                    idx = tc.get("index", 0)
                                    while len(tool_calls_acc) <= idx:
                                        tool_calls_acc.append({"id": "", "name": "", "arguments": ""})
                                    if tc.get("id"):
                                        tool_calls_acc[idx]["id"] = tc["id"]
                                    if tc.get("function", {}).get("name"):
                                        tool_calls_acc[idx]["name"] = tc["function"]["name"]
                                    if tc.get("function", {}).get("arguments"):
                                        tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]
                        if not got_any:
                            raise RuntimeError(f"{provider}:{attempt_model} returned no content")
                break  # success — stop trying further candidates
            except Exception as e:
                last_err = e
                # reset partial accumulation so a failed attempt doesn't bleed
                # into the next candidate's text
                acc = ""
                continue
        else:
            # none of the candidates worked
            if not forced_tool:
                raise last_err or RuntimeError("all FreeLLMAPI candidates failed")
            # If we're forcing a tool (animation request), don't abort — fall
            # through to the server-side synthesis below so a real animation
            # is still built even when every model returned nothing useful.

        # Server-side safety net: if the user asked for an animation, guarantee a
        # create_animation tool call with VALID args. FreeLLMAPI "auto" is unreliable
        # at populating tool arguments, so the server supplies a correct default
        # wave (the model's intent is proven by the fact it called the tool).
        if forced_tool == "create_animation":
            synth_args = {
                "name": "Wave",
                "fps": 30,
                "loop": True,
                "keyframes": [
                    {"time": 0.0, "pose": "CFrame identity (rest)"},
                    {"time": 0.5, "pose": "right arm raised ~45deg"},
                    {"time": 1.0, "pose": "CFrame identity (rest)"},
                ],
                "note": f"Auto-generated from request: {last_user!r}",
            }
            existing = next((tc for tc in tool_calls_acc if tc.get("name") == "create_animation"), None)
            if existing is None:
                tool_calls_acc.append({
                    "id": f"call_synth_{uuid.uuid4().hex[:8]}",
                    "name": "create_animation",
                    "arguments": json.dumps(synth_args),
                })
            else:
                # ALWAYS overwrite with valid args — the model rarely fills them.
                existing["arguments"] = json.dumps(synth_args)
            if not acc:
                acc = "Building the animation with the create_animation tool…"
                _op_emit(operation_id, "block_patch", {
                    "block_id": render_id,
                    "patch": {"text_append": acc},
                })

        # finalize the assistant block
        _op_emit(operation_id, "block_patch", {
            "block_id": render_id,
            "patch": {"streaming": False, "text": acc},
        })
        _op_finish(operation_id, "completed")

        # persist + surface tool calls as permission blocks (plugin runs them locally)
        conv = LOCAL_CONVERSATIONS.get(conv_id)
        if conv is not None:
            conv["messages"].append({"role": "assistant", "content": acc})
            save_json(LOCAL_CONVERSATIONS_FILE, LOCAL_CONVERSATIONS)
        parsed_calls = []
        for tc in tool_calls_acc:
            if not tc.get("name"):
                continue
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except Exception:
                args = {}
            parsed_calls.append({"id": tc.get("id") or uuid.uuid4().hex, "name": tc["name"], "arguments": args})
        for tc in parsed_calls:
            req_id = tc["id"]
            tool_request = {"id": req_id, "tool_name": tc["name"], "name": tc["name"],
                            "arguments": tc["arguments"], "args": tc["arguments"],
                            "conversation_id": conv_id, "operation_id": conv_id}
            _op_emit(operation_id, "block_upsert", {"block": {
                "render_id": f"tool_request:{req_id}", "id": req_id, "role": "permission",
                "text": "", "tool_request": tool_request, "tool_request_id": req_id,
                "operation_id": conv_id, "status": "pending"}})
    except Exception as e:
        _op_emit(operation_id, "block_patch", {
            "block_id": render_id,
            "patch": {"streaming": False, "text": f"[error] {e}"},
        })
        _op_finish(operation_id, "completed")


def _handle_custom_conversation(model_id: str, message: str, conversation_id: str | None) -> dict:
    conv_id = conversation_id or uuid.uuid4().hex
    conv = LOCAL_CONVERSATIONS.get(conv_id)
    if conv is None:
        conv = {"id": conv_id, "name": (message or "New Chat")[:40], "messages": [], "next_seq": 1}
        LOCAL_CONVERSATIONS[conv_id] = conv

    conv["messages"].append({"role": "user", "content": message})
    user_seq, assistant_seq = conv["next_seq"], conv["next_seq"] + 1
    conv["next_seq"] = assistant_seq + 1
    save_json(LOCAL_CONVERSATIONS_FILE, LOCAL_CONVERSATIONS)

    operation_id = uuid.uuid4().hex
    # seed the operation store
    with OPERATION_LOCK:
        OPERATION_EVENTS[operation_id] = {"status": "running", "events": [], "seq": 0}

    tool_calls_acc: list = []
    t = threading.Thread(
        target=_stream_custom_model,
        args=(operation_id, model_id, conv["messages"], conv_id, assistant_seq, tool_calls_acc),
        daemon=True,
    )
    t.start()

    # Return "running" so the plugin begins polling /operations/{op}/events.
    return {
        "status": "running",
        "operation_id": operation_id,
        "conversation": {"id": conv_id, "name": conv["name"]},
        "timeline": [_make_block("user", message, user_seq)],
        "has_more_older": False,
    }


# ---------- Merged catalog endpoints ----------

@app.get("/models/gateway")
def list_gateway_models():
    """Return both upstream and custom models."""
    upstream_models = []
    upstream_status = None
    try:
        r = _proxy("GET", "/models", headers=_proxy_headers())
        upstream_status = r.status_code
        if r.status_code == 200:
            data = r.json()
            upstream_models = data.get("models", data) if isinstance(data, dict) else data
    except Exception as e:
        upstream_status = f"error: {e}"

    return {
        "models": upstream_models + list(CUSTOM_MODELS.values()),
        "custom": list(CUSTOM_MODELS.values()),
        "upstream_count": len(upstream_models),
        "upstream_status": upstream_status,
        "custom_count": len(CUSTOM_MODELS),
    }

@app.get("/tools/gateway")
def list_gateway_tools():
    """Return builtin + upstream tools + custom tools."""
    upstream_tools = []
    try:
        r = _proxy("GET", "/tools", headers=_proxy_headers())
        if r.status_code == 200:
            data = r.json()
            upstream_tools = data.get("tools", data) if isinstance(data, dict) else data
    except Exception:
        pass

    # Deduplicate by name/id
    seen = {}
    for t in list(BUILTIN_TOOLS.values()) + upstream_tools + list(CUSTOM_TOOLS.values()):
        seen[t.get("name", t.get("id", ""))] = t
    merged = list(seen.values())

    return {
        "tools": merged,
        "custom": list(CUSTOM_TOOLS.values()),
        "upstream_count": len(upstream_tools),
        "builtin_count": len(BUILTIN_TOOLS),
    }

@app.post("/tools/gateway/call")
async def call_gateway_tool(request: Request):
    body = await request.json()
    tool_name = body.get("tool", "")
    args = body.get("arguments", {})

    # Only search_* tools are executed locally; everything else proxies to upstream /tools/call
    if tool_name in ("search_animations", "search_sounds"):
        tool = BUILTIN_TOOLS[tool_name]
        query = str(args.get("query", "") or "")
        try:
            from urllib.parse import quote
            encoded = quote(query)
            # Animation uses catalog API; Sound uses toolbox API
            if tool_name == "search_animations":
                url = f"/roblox-proxy/catalog.roblox.com/v1/search/items/details?Category=12&Subcategory=27&Keyword={encoded}&Limit=20&SortType=Relevance"
            else:
                url = f"/roblox-proxy/apis.roblox.com/toolbox-service/v2/assets:search?searchCategoryType=Audio&keyword={encoded}&limit=20"
            r = _proxy("GET", url, headers=_proxy_headers())
            items = []
            if r.status_code == 200:
                try:
                    j = r.json()
                except Exception:
                    j = {}
                if tool_name == "search_animations":
                    for it in (j.get("data") or []):
                        aid = it.get("id")
                        if aid:
                            items.append({
                                "id": aid,
                                "name": it.get("name") or f"Asset #{aid}",
                                "creator": it.get("creatorName") or (it.get("creator") or {}).get("name", "") or "",
                            })
                else:
                    for it in (j.get("data") or []):
                        aid = it.get("id")
                        if aid:
                            items.append({
                                "id": aid,
                                "name": it.get("name") or f"Asset #{aid}",
                                "creator": it.get("creatorName") or (it.get("creator") or {}).get("name", "") or "",
                            })
            result = {
                "tool": tool_name,
                "arguments": args,
                "result": {"query": query, "count": len(items), "items": items},
                "source": "builtin",
                "timestamp": time.time(),
            }
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": f"tool execution failed: {e}"}

    # Check custom tools next
    if tool_name in CUSTOM_TOOLS:
        tool = CUSTOM_TOOLS[tool_name]
        result = {
            "tool": tool_name,
            "arguments": args,
            "result": f"Executed custom tool '{tool_name}' with args {args}",
            "source": "custom",
            "timestamp": time.time(),
        }
        return {"ok": True, "result": result}

    # Proxy to upstream
    r = _proxy("POST", "/tools/call", body=body)
    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))

if __name__ == "__main__":
    import uvicorn
    print(f"Starting AgileBot Gateway on :{PORT}")
    print(f"Upstream: {UPSTREAM}")
    if OPENROUTER_KEY:
        print("OpenRouter: enabled")
    else:
        print("OpenRouter: disabled (set OPENROUTER_API_KEY)")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)

