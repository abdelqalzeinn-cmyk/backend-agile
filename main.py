"""AgileBot Gateway Backend
- Proxies to https://api.agilebot.dev for native models/tools
- Adds custom models + tools on top
- Routes custom models to OpenRouter or local providers
- Integrates model broker communication with https://broker-for-model-agile.onrender.com
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
BROKER_URL = os.environ.get("BROKER_URL", "https://broker-for-model-agile.onrender.com")
PORT = int(os.environ.get("PORT", "8765"))
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BUILD_TAG = "build-2026-08-01-createanim-synth-v3"
FREELLMAPI_URL = os.environ.get("FREELLMAPI_URL", "https://freellmapi-cliz.onrender.com")
FREELLMAPI_KEY = os.environ.get("FREELLMAPI_KEY", "")

app = FastAPI(title="AgileBot Gateway", version="0.3.0")

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

OPERATION_EVENTS: dict = {}
OPERATION_LOCK = threading.Lock()

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

class UpstreamUnavailable(Exception):
    def __init__(self, message: str, url: str):
        super().__init__(message)
        self.message = message
        self.url = url

def _proxy(method: str, path: str, body: dict | bytes | None = None, headers: dict | None = None) -> httpx.Response:
    url = f"{UPSTREAM}{path}"
    merged = dict(_proxy_headers())
    if headers:
        merged.update(headers)
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            if method == "GET":
                return client.get(url, headers=merged)
            if isinstance(body, bytes):
                return client.request(method, url, content=body, headers=merged)
            return client.request(method, url, json=body, headers=merged)
    except httpx.RequestError as e:
        # Surface upstream connectivity failures as a real HTTP response instead of
        # letting the exception propagate and kill the connection with nothing sent
        # back to the client. A silent connection reset means Roblox's HttpService
        # never gets a response at all, so its own error logging never fires.
        print(f"[PROXY] Failed to reach upstream {url}: {e}")
        raise UpstreamUnavailable(str(e), url) from e

def _safe_proxy(method: str, path: str, body=None, headers=None):
    """Like _proxy, but returns (Response|None, error_json|None) instead of raising."""
    try:
        r = _proxy(method, path, body=body, headers=headers)
        return r, None
    except UpstreamUnavailable as e:
        return None, {
            "detail": {
                "code": "upstream_unreachable",
                "message": f"Could not reach {UPSTREAM}: {e.message}",
            }
        }

# ---------- Health ----------

@app.get("/health")
def health():
    return {
        "ok": True,
        "upstream": UPSTREAM,
        "broker": BROKER_URL,
        "build_tag": BUILD_TAG,
        "custom_models": len(CUSTOM_MODELS),
        "custom_tools": len(CUSTOM_TOOLS),
    }

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

# ---------- Broker Integration Routes ----------

@app.post("/api/agent/chat")
async def proxy_to_broker_chat(request: Request):
    """Relays agent task submissions to the broker service."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    async with httpx.Client(timeout=30.0) as client:
        try:
            resp = await client.post(f"{BROKER_URL.rstrip('/')}/api/agent/chat", json=body)
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        except Exception as e:
            return JSONResponse({"error": f"Failed to connect to broker: {e}"}, status_code=502)

@app.get("/api/agent/status/{job_id}")
async def proxy_to_broker_status(job_id: str):
    """Polls job status from the broker service."""
    async with httpx.Client(timeout=30.0) as client:
        try:
            resp = await client.get(f"{BROKER_URL.rstrip('/')}/api/agent/status/{job_id}")
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        except Exception as e:
            return JSONResponse({"error": f"Failed to fetch job status from broker: {e}"}, status_code=502)

# ---------- Workspace & Conversations ----------

@app.get("/workspace")
async def workspace(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    r, err = _safe_proxy("GET", "/workspace", headers=h)
    if err:
        return JSONResponse(content=err, status_code=502)
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
    r, err = _safe_proxy("GET", "/conversations", headers=h)
    if err:
        return JSONResponse(content=err, status_code=502)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))

@app.post("/conversations")
async def conversations_post(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    
    try:
        body_json = await request.json()
    except Exception:
        body_json = {}

    if isinstance(body_json, str):
        try:
            body_json = json.loads(body_json)
        except Exception:
            body_json = {"message": body_json}

    model_id = str(body_json.get("model", "auto"))
    if model_id == "auto" or not model_id:
        model_id = "freellmapi/auto"
        body_json["model"] = model_id

    message_text = str(body_json.get("message", "") or body_json.get("text", ""))
    if not message_text and isinstance(body_json.get("context_revision"), dict):
        rev = body_json.get("context_revision")
        message_text = str(rev.get("message", "") or rev.get("prompt", ""))

    # Intercept and create the conversation locally so it never fails upstream
    if _is_custom_model(model_id):
        try:
            result = _handle_custom_conversation(model_id, message_text, None)
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            return JSONResponse({"detail": {"code": "custom_model_error", "model": model_id, "message": str(e)}}, status_code=502)

    r, err = _safe_proxy("POST", "/conversations", body=body_json, headers=h)
    if err:
        return JSONResponse(content=err, status_code=502)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))

@app.post("/conversations/{conversation_id}/messages")
async def conversations_messages_post(conversation_id: str, request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    
    # Read the raw body safely as text first to prevent JSON decode crashes
    body_bytes = await request.body()
    body_json = {}
    
    if body_bytes:
        try:
            body_json = json.loads(body_bytes.decode('utf-8'))
        except Exception:
            # If it's a raw string payload, wrap it into a message dictionary
            body_json = {"message": body_bytes.decode('utf-8', errors='ignore')}

    # If body_json ended up as a string or non-dict, normalize it
    if isinstance(body_json, str):
        try:
            body_json = json.loads(body_json)
        except Exception:
            body_json = {"message": body_json}

    model_id = str(body_json.get("model", "auto"))
    if model_id == "auto" or not model_id:
        model_id = "freellmapi/auto"

    message_text = str(body_json.get("message", "") or body_json.get("text", ""))
    if not message_text and isinstance(body_json.get("context_revision"), dict):
        rev = body_json.get("context_revision")
        message_text = str(rev.get("message", "") or rev.get("prompt", ""))

    if _is_custom_model(model_id) or conversation_id in LOCAL_CONVERSATIONS:
        try:
            result = _handle_custom_conversation(model_id, message_text, conversation_id)
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            return JSONResponse({"detail": {"code": "custom_model_error", "model": model_id, "message": str(e)}}, status_code=502)

    r, err = _safe_proxy("POST", f"/conversations/{conversation_id}/messages", body=body_json, headers=h)
    if err:
        return JSONResponse(content=err, status_code=502)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))

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

# ---------- FreeLLMAPI Sync & Forwarding ----------

def _freellmapi_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if FREELLMAPI_KEY:
        h["Authorization"] = f"Bearer {FREELLMAPI_KEY}"
    return h

def fetch_freellmapi_models() -> list:
    if not FREELLMAPI_URL:
        return []
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
            r = client.get(f"{FREELLMAPI_URL.rstrip('/')}/v1/models", headers=_freellmapi_headers())
            if r.status_code == 200:
                data = r.json()
                return data.get("data", data) if isinstance(data, dict) else data
    except Exception:
        pass
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
    return {"added": added, "total": len(models)}

def forward_freellmapi(body: dict) -> Response:
    url = f"{FREELLMAPI_URL.rstrip('/')}/v1/chat/completions"
    headers = _freellmapi_headers()
    headers["HTTP-Referer"] = "http://localhost:8765"
    headers["X-Title"] = "AgileBot Gateway"
    fixed_body = dict(body)
    fixed_body["model"] = "auto"
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        r = client.post(url, json=fixed_body, headers=headers)
    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))

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

def fetch_openrouter_free_models():
    if not OPENROUTER_KEY:
        return []
    try:
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
            r = client.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {OPENROUTER_KEY}"})
            if r.status_code == 200:
                data = r.json()
                return [m for m in data.get("data", []) if ":free" in m.get("id", "")]
    except Exception:
        pass
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
                "endpoint": "https://openrouter.ai/api/v1",
                "context_length": m.get("context_length", 8192),
                "created_at": time.time(),
            }
            added.append(mid)
    if added:
        save_json(CUSTOM_MODELS_FILE, CUSTOM_MODELS)
    return {"added": added}

def _is_custom_model(model_id: str) -> bool:
    if not model_id:
        return False
    # Treat "auto", aliases, and any freellmapi/openrouter prefix as a custom model
    if model_id in CUSTOM_MODELS or model_id in CUSTOM_MODEL_ALIASES or model_id == "auto":
        return True
    if model_id.startswith("freellmapi/") or model_id.startswith("openrouter/"):
        return True
    return False
    
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
        
def _make_block(role: str, text: str, seq_n: int) -> dict:
    rid = f"message:{seq_n}"
    return {
        "id": rid,
        "render_id": rid,
        "role": role,
        "text": text,
        "seq": seq_n,
        "created_at_unix_ms": int(time.time() * 1000),
    }

def _op_emit(operation_id: str, event_type: str, payload: dict):
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

def _stream_custom_model(operation_id: str, model_id: str, messages: list, conv_id: str,
                         assistant_seq: int, tool_calls_acc: list):
    """Background worker: call the provider with stream:true, push block events
    to the operation store so the plugin renders live typing."""
    render_id = f"message:{assistant_seq}"
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
        url = f"{FREELLMAPI_URL.rstrip('/')}/v1/chat/completions"
        headers = _freellmapi_headers()
        headers["HTTP-Referer"] = "http://localhost:8765"
        headers["X-Title"] = "AgileBot Gateway"

        sys_msgs = list(messages)
        if SYSTEM_PROMPT:
            if not sys_msgs or sys_msgs[0].get("role") != "system":
                sys_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + sys_msgs
            else:
                sys_msgs[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + sys_msgs[0].get("content", "")}

        body = {"model": "auto", "messages": sys_msgs, "stream": True}
        
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Stream error {resp.status_code}: {resp.read().decode()[:200]}")
                for chunk in resp.iter_lines():
                    for data in _parse_sse_lines(chunk):
                        try:
                            piece = json.loads(data)
                        except Exception:
                            continue
                        delta = (piece.get("choices") or [{}])[0].get("delta") or {}
                        text_delta = delta.get("content") or ""
                        if text_delta:
                            acc += text_delta
                            _op_emit(operation_id, "block_patch", {
                                "block_id": render_id,
                                "patch": {"text_append": text_delta},
                            })

        _op_emit(operation_id, "block_patch", {
            "block_id": render_id,
            "patch": {"streaming": False, "text": acc},
        })
        _op_finish(operation_id, "completed")

        conv = LOCAL_CONVERSATIONS.get(conv_id)
        if conv is not None:
            conv["messages"].append({"role": "assistant", "content": acc})
            save_json(LOCAL_CONVERSATIONS_FILE, LOCAL_CONVERSATIONS)

    except Exception as e:
        _op_emit(operation_id, "block_patch", {
            "block_id": render_id,
            "patch": {"streaming": False, "text": f"[error] {e}"},
        })
        _op_finish(operation_id, "completed")

def _handle_custom_conversation(model_id: str, message: str, conversation_id: str | None) -> dict:
    conv_id = conversation_id or uuid.uuid4().hex
    conv = LOCAL_CONVERSATIONS.get(conv_id)
    now_ms = int(time.time() * 1000)
    if conv is None:
        conv = {
            "id": conv_id, 
            "name": (message or "New Chat")[:40], 
            "messages": [], 
            "next_seq": 1,
            "created_at": now_ms
        }
        LOCAL_CONVERSATIONS[conv_id] = conv

    # Guard against duplicate user messages
    last = conv["messages"][-1] if conv["messages"] else None
    dup = (last and last.get("role") == "user" and last.get("content") == message
           and (now_ms - int(last.get("_ts", 0))) < 2000)
    if not dup and message:
        conv["messages"].append({"role": "user", "content": message, "_ts": now_ms})
    
    user_seq = conv["next_seq"]
    assistant_seq = user_seq + 1
    conv["next_seq"] = assistant_seq + 1
    save_json(LOCAL_CONVERSATIONS_FILE, LOCAL_CONVERSATIONS)

    operation_id = uuid.uuid4().hex
    conv["_active_operation_id"] = operation_id
    
    with OPERATION_LOCK:
        OPERATION_EVENTS[operation_id] = {"status": "running", "events": [], "seq": 0}

    tool_calls_acc: list = []
    t = threading.Thread(
        target=_stream_custom_model,
        args=(operation_id, model_id, conv["messages"], conv_id, assistant_seq, tool_calls_acc),
        daemon=True,
    )
    t.start()

    # Return a structured response matching normal conversation creation objects
    return {
        "id": conv_id,
        "name": conv["name"],
        "status": "running",
        "operation_id": operation_id,
        "conversation": {
            "id": conv_id,
            "name": conv["name"],
            "created_at": now_ms
        },
        "timeline": [_make_block("user", message, user_seq)] if message else [],
        "has_more_older": False,
    }

def list_gateway_tools():
    return {"tools": list(BUILTIN_TOOLS.values())}

def call_gateway_tool(request: Request):
    return {"ok": True}

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
    body = await request.body()
    r = _proxy("POST", f"/{path}", body=body, headers=h)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
