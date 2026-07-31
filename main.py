"""AgileBot Gateway Backend
- Proxies to https://api.agilebot.dev for native models/tools
- Adds custom models + tools on top
- Routes custom models to OpenRouter or local providers
"""

import os
import json
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import httpx

UPSTREAM = os.environ.get("AGILEBOT_UPSTREAM", "https://api.agilebot.dev")
PORT = int(os.environ.get("PORT", "8765"))
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

app = FastAPI(title="AgileBot Gateway", version="0.2.0")

# ---------- Persistence ----------

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CUSTOM_TOOLS_FILE = DATA_DIR / "custom_tools.json"
CUSTOM_MODELS_FILE = DATA_DIR / "custom_models.json"

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

CUSTOM_TOOLS: dict = load_json(CUSTOM_TOOLS_FILE, {})
CUSTOM_MODELS: dict = load_json(CUSTOM_MODELS_FILE, {})

# ---------- Built-in tools ----------

BUILTIN_TOOLS = {
    "search_animations": {
        "name": "search_animations",
        "description": "Search Roblox catalog for animations by keyword. Returns asset id, name, creator.",
        "parameters": {
            "query": "string - search keyword",
            "category": "string - optional filter (e.g. emote, run, idle)",
        },
        "endpoint": "/roblox-proxy/apis.roblox.com/toolbox-service/v1/search?query={query}&category=Animation",
    },
    "search_sounds": {
        "name": "search_sounds",
        "description": "Search Roblox catalog for sounds/audio by keyword.",
        "parameters": {
            "query": "string - search keyword",
        },
        "endpoint": "/roblox-proxy/apis.roblox.com/toolbox-service/v1/search?query={query}&category=Audio",
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
        "user-agent": "AgileBotGateway/0.1 (+https://github.com/abdelqalzeinn-cmyk/backend-agile)",
    }

def _proxy(method: str, path: str, body: dict | None = None, headers: dict | None = None) -> httpx.Response:
    url = f"{UPSTREAM}{path}"
    merged = dict(_proxy_headers())
    if headers:
        merged.update(headers)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        if method == "GET":
            return client.get(url, headers=merged)
        return client.request(method, url, json=body, headers=merged)

# ---------- Health ----------

@app.get("/health")
def health():
    return {
        "ok": True,
        "upstream": UPSTREAM,
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
    result = sync_openrouter_free_models()
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
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))
    try:
        data = r.json()
        return JSONResponse(content=data, status_code=200, headers={"Content-Encoding": "identity"})
    except Exception:
        return Response(content=r.content, status_code=r.status_code, headers={"Content-Encoding": "identity"})


@app.post("/conversations")
async def conversations_post(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    body = await request.body()
    r = _proxy("POST", "/conversations", body=body, headers=h)
    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))


@app.get("/{path:path}")
async def proxy_get(path: str, request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    r = _proxy("GET", f"/{path}", headers=h)
    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))

@app.post("/{path:path}")
async def proxy_post(path: str, request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
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
                        body["tools"].append(t)
                # lightweight system hint unless one already set
                msgs = body.get("messages") or []
                hint = "You may use gateway tools:" + ", ".join(t["name"] for t in formatted)
                if not any(isinstance(m, dict) and m.get("role") == "system" for m in msgs):
                    msgs = [{"role": "system", "content": hint}] + msgs
                    body["messages"] = msgs

        # Route custom models to their provider
        if model_id in CUSTOM_MODELS:
            model = CUSTOM_MODELS[model_id]
            provider = model.get("provider", "openrouter")
            endpoint = model.get("endpoint", "https://openrouter.ai/api/v1")

            if provider == "openrouter":
                if not OPENROUTER_KEY:
                    return JSONResponse({"error": "OPENROUTER_API_KEY not set"}, status_code=500)
                return forward_openrouter(endpoint, body)

            if provider == "local":
                return forward_local(body)

            return JSONResponse({"error": f"Unknown provider: {provider}"}, status_code=400)

        # Not a custom model — proxy to upstream
        r = _proxy("POST", f"/{path}", body=body, headers=_add_auth(dict(_proxy_headers()), request))
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))

    # Default: proxy everything else
    try:
        body = await request.json()
    except Exception:
        body = {}
    h = dict(_proxy_headers())
    r = _proxy("POST", f"/{path}", body=body, headers=_add_auth(h, request))
    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))

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
        "sample": [m.get("id") for m in fetched[:5]],
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

    # Check builtin tools first
    if tool_name in BUILTIN_TOOLS:
        tool = BUILTIN_TOOLS[tool_name]
        result = {
            "tool": tool_name,
            "arguments": args,
            "result": f"Executed builtin tool '{tool_name}' with args {args}",
            "source": "builtin",
            "timestamp": time.time(),
        }
        return {"ok": True, "result": result}

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


@app.post("/admin/system-prompt")
async def update_system_prompt(request: Request):
    """Inject custom instructions into the system prompt."""
    body = await request.json()
    instruction = body.get("instruction", "")
    if not instruction:
        return {"ok": False, "error": "instruction required"}
    return {
        "ok": True,
        "system_prompt": instruction,
        "available_tools": list(BUILTIN_TOOLS.keys()) + list(CUSTOM_TOOLS.keys()),
        "message": "System prompt updated. Restart backend or reload config to apply."
    }

@app.get("/admin/system-prompt")
async def get_system_prompt():
    """Get current system prompt and available tools."""
    return {
        "available_tools": list(BUILTIN_TOOLS.keys()) + list(CUSTOM_TOOLS.keys()),
        "builtin_count": len(BUILTIN_TOOLS),
        "custom_count": len(CUSTOM_TOOLS),
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting AgileBot Gateway on :{PORT}")
    print(f"Upstream: {UPSTREAM}")
    if OPENROUTER_KEY:
        print("OpenRouter: enabled")
    else:
        print("OpenRouter: disabled (set OPENROUTER_API_KEY)")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)

