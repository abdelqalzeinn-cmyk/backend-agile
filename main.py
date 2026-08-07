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
import asyncio
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
    "You are AgileBot, an expert Roblox Studio AI assistant and developer that writes clean, correct Luau code.\n\n"

    "TOOLS\n"
    "- create_animation, search_animations, search_sounds: run INSIDE Roblox Studio.\n"
    "- web_search: live web search for up-to-date docs/facts, executed via a Colab worker. Results can take a "
    "few seconds — do not repeat the same query while waiting.\n"
    "- run_cmds: runs shell commands/scripts in the Colab VM, not in Roblox Studio. Use it for setup, file work, "
    "or anything that needs a real shell — never for Roblox-side actions.\n\n"

    "CORE LOOP\n"
    "- The latest user message is the objective. Earlier conversation is context only and may be stale.\n"
    "- When the user asks you to build, animate, search, look up live information, or run a command, you MUST "
    "call the matching tool via a tool_call. NEVER just describe the steps, NEVER hand back code or commands for "
    "the user to run manually, and NEVER say you can't access the web/terminal — you can, via these tools.\n"
    "- When you call a tool, populate EVERY relevant parameter with a concrete, real value. Never call a tool "
    "with empty, placeholder, or guessed-blank arguments.\n"
    "- Only fall back to writing Luau in chat when the user explicitly asks for raw script text, or after a tool "
    "result gives you the concrete facts (e.g. search results, rig type) you need to write it correctly.\n"
    "- Prefer action over narration: don't announce what you're about to do, just do it, then report the result.\n"
    "- Do not repeat an identical tool call expecting a different result. If a search/command comes back weak or "
    "empty, change the query/approach rather than retrying verbatim.\n"
    "- Stop once the request is satisfied. Do not keep adding unrequested changes or extra tool calls.\n"
    "- Never describe unfinished or unverified work as complete. Only claim success if the tool result actually "
    "shows it succeeded (e.g. run_cmds exit code, search actually returned results).\n"
    "- If a tool result shows an error, fix or report the specific failure — do not silently retry blind or "
    "pretend it worked.\n\n"

    "READ-ONLY VS ACTION\n"
    "- Questions, explanations, and diagnosis (\"what's wrong with this script\", \"how does X work\") are answered "
    "directly or via web_search — do not build/animate/execute anything the user didn't ask for.\n"
    "- Only use create_animation/search_animations/search_sounds/run_cmds when the user's request actually "
    "requires changing or running something.\n"
    "- Don't turn a diagnosis request into an unrequested fix. Don't turn a \"what's the latest version\" question "
    "into a code change.\n\n"

    "RIG / ANIMATION COMPATIBILITY\n"
    "- R6 and R15 animations are not interchangeable. Never apply an R15 animation to an R6 rig or vice versa.\n"
    "- Before creating, replacing, or wiring any character animation, confirm the target rig type from live "
    "evidence (e.g. via search_animations results or explicit user statement) — never infer rig type from names "
    "or appearance alone.\n"
    "- If rig type can't be established and the user hasn't stated it, ask one concise clarifying question rather "
    "than guessing.\n"
    "- Prefer using an existing AnimationId over hand-rolled custom animating when one fits.\n\n"

    "LUAU CODE STANDARDS\n"
    "- Follow Roblox Luau conventions: use task.spawn/task.wait instead of the deprecated spawn/wait.\n"
    "- Guard pcall around HttpService and DataStore calls.\n"
    "- Never use Lua 5.1-only syntax (no +=, no const, etc. that Luau doesn't support in this context).\n"
    "- Keep scripts small, readable, and production-safe — no dead code, no giant unexplained blocks.\n\n"

    "RESPONSE STYLE\n"
    "- Keep final answers concise and user-facing; skip internal mechanics unless asked.\n"
    "- Do not output fake progress updates, promise future work, or invent blockers.\n"
    "- Do not claim a build/search/command succeeded unless the tool result proves it.\n"
    "- Only use emojis if the user explicitly asks for them."
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
    "scene_overview": {
        "name": "scene_overview",
        "description": "Get a high-level layout of the current place: top-level services, major landmarks, and rough object-count signature per subtree. Orientation only, not an exact index — follow up with instance_find for exact targets.",
        "parameters": {
            "type": "object",
            "properties": {
                "scope_path": {"type": "string", "description": "Optional subtree root to limit the overview to (e.g. 'Workspace')."}
            },
            "required": [],
        },
    },
    "instance_find": {
        "name": "instance_find",
        "description": "Search Roblox instances by name/path in the live place. Use scope_path to limit to one subtree and class_name to narrow by ClassName.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Instance name or partial name to search for."},
                "scope_path": {"type": "string", "description": "Optional subtree root to search within."},
                "class_name": {"type": "string", "description": "Optional ClassName filter, e.g. 'Part', 'Script'."}
            },
            "required": ["query"],
        },
    },
    "instance_inspect": {
        "name": "instance_inspect",
        "description": "Get exact properties/attributes/children of one known Instance. Use only when you already have an exact ref or path — use instance_find first if the target is unknown.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Instance ref, if known from a prior tool result."},
                "path": {"type": "string", "description": "Full workspace/hierarchy path to the instance."}
            },
            "required": [],
        },
    },
    "script_inventory": {
        "name": "script_inventory",
        "description": "List all Script/LocalScript/ModuleScript instances in the place (or a subtree), with path and basic size info. Use for broad discovery before searching contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "scope_path": {"type": "string", "description": "Optional subtree root to limit the inventory to."}
            },
            "required": [],
        },
    },
    "script_find": {
        "name": "script_find",
        "description": "Find scripts by name/path pattern. Use before script_read/script_grep when the exact script path is unknown.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Script name or partial name to search for."},
                "scope_path": {"type": "string", "description": "Optional subtree root to search within."}
            },
            "required": ["query"],
        },
    },
    "script_grep": {
        "name": "script_grep",
        "description": "Search script source code across the place (or a subtree) for a literal string or pattern. Returns matching paths and line context.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Literal text or pattern to search for in script source."},
                "scope_path": {"type": "string", "description": "Optional subtree root to limit the search to."},
                "case_sensitive": {"type": "boolean", "description": "Whether the match is case-sensitive (default false)."}
            },
            "required": ["pattern"],
        },
    },
    "script_read": {
        "name": "script_read",
        "description": "Read the exact current source of one known script (by ref or path), optionally a line range. Always use this to get the real source_hash/raw_source before writing or replacing.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Script ref, if known from a prior tool result."},
                "path": {"type": "string", "description": "Full path to the script instance."},
                "start_line": {"type": "number", "description": "Optional starting line (1-indexed)."},
                "end_line": {"type": "number", "description": "Optional ending line (inclusive)."}
            },
            "required": [],
        },
    },
    "script_write": {
        "name": "script_write",
        "description": "Overwrite the ENTIRE source of one script with new content. Use for creating a new script or a full rewrite; use script_replace for a small targeted edit instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Script ref, if known from a prior tool result."},
                "path": {"type": "string", "description": "Full path to the script instance (created if it doesn't exist)."},
                "class_name": {"type": "string", "description": "Script class to create if new: 'Script', 'LocalScript', or 'ModuleScript'."},
                "source": {"type": "string", "description": "Complete new source code for the script."}
            },
            "required": ["source"],
        },
    },
    "script_write_many": {
        "name": "script_write_many",
        "description": "Overwrite the entire source of multiple scripts in one batched call. Each entry behaves like script_write.",
        "parameters": {
            "type": "object",
            "properties": {
                "writes": {
                    "type": "array",
                    "description": "List of full-source writes to apply.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref": {"type": "string"},
                            "path": {"type": "string"},
                            "class_name": {"type": "string"},
                            "source": {"type": "string"}
                        },
                        "required": ["source"]
                    }
                }
            },
            "required": ["writes"],
        },
    },
    "script_replace": {
        "name": "script_replace",
        "description": "Apply one exact find/replace edit to a script's existing source. old_string must be an exact, unique substring of the script's CURRENT source (re-read with script_read first if unsure) — do not guess it.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Script ref, if known from a prior tool result."},
                "path": {"type": "string", "description": "Full path to the script instance."},
                "old_string": {"type": "string", "description": "Exact, unique substring of the current source to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "source_hash": {"type": "string", "description": "Hash of the source this edit was computed against, from the latest script_read, to guard against stale edits."}
            },
            "required": ["old_string", "new_string"],
        },
    },
    "script_replace_many": {
        "name": "script_replace_many",
        "description": "Apply multiple exact find/replace edits to one script atomically. All edits succeed together or the script is left unchanged.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"}
                        },
                        "required": ["old_string", "new_string"]
                    }
                },
                "source_hash": {"type": "string", "description": "Hash of the source these edits were computed against."}
            },
            "required": ["edits"],
        },
    },
    "verify_changes": {
        "name": "verify_changes",
        "description": "Re-read a script/instance after an edit and confirm the change actually applied as intended. Use after script_write/script_replace before claiming success.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "path": {"type": "string"},
                "expected_hash": {"type": "string", "description": "Optional expected source_hash to confirm against."}
            },
            "required": [],
        },
    },
    "execute_lua": {
        "name": "execute_lua",
        "description": "Run a small, focused Luau snippet in Studio for things no other tool covers: creating non-script Instances, runtime state/property probes, aggregation/counting across many objects, or geometry computation. NOT for authoring script source (use script_write/script_replace). Must explicitly return the values needed; do not rely on print. Keep snippets small and purpose-specific, not a dump.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Luau source to execute. Must end with an explicit return of the needed value(s)."}
            },
            "required": ["code"],
        },
    },
    "startup_smoke_test": {
        "name": "startup_smoke_test",
        "description": "Run the place's startup scripts in a fresh simulated start to confirm nothing errors. Use after larger changes when startup correctness matters, not for every small edit.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the live web for up-to-date documentation, APIs, facts, or technical solutions. Use this for real-time information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to perform."}
            },
            "required": ["query"],
        },
        "endpoint": "https://broker-for-model-agile.onrender.com/tools/web_search?query={query}",
    },
    "run_cmds": {
        "name": "run_cmds",
        "description": "Execute system terminal commands, environment setup scripts, or automation tasks via the broker.",
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {"type": "string", "description": "The specific command or script string to execute."}
            },
            "required": ["commands"],
        },
        "endpoint": "https://broker-for-model-agile.onrender.com/tools/run_cmds",
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
def tools_gateway_call(tool_name: str, args: dict, conv_id: str):
    """Proxies tool execution requests to the broker."""
    try:
        url = f"{BROKER_URL.rstrip('/')}/tools/execute"
        payload = {
            "tool_name": tool_name,
            "arguments": args,
            "conversation_id": conv_id
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                return {"ok": False, "error": f"Broker tool error: {resp.status_code}"}
            return {"ok": True, "result": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
@app.get("/conversations/{conversation_id}/timeline")
def conversations_timeline(conversation_id: str, request: Request, operation_id: str = None):
    conv = LOCAL_CONVERSATIONS.get(conversation_id)
    if conv is None:
        return JSONResponse(content={"conversation": {"id": conversation_id, "name": "Custom Session"}, "timeline": [], "has_more_older": False}, status_code=200)

    blocks = []
    seq = 1
    for m in conv.get("messages", []):
        role = m.get("role", "user")
        blocks.append(_make_block(role, m.get("content", ""), seq))
        seq += 1
        
    return JSONResponse(content={
        "conversation": {"id": conv.get("id", conversation_id), "name": conv.get("name", "Session")},
        "timeline": blocks,
        "has_more_older": False,
    })

@app.post("/operations/{operation_id}/tool_results")
async def operations_tool_results_post(operation_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Best-effort forward to broker (kept for logging/compat), failures ignored
    try:
        url = f"{BROKER_URL.rstrip('/')}/operations/{operation_id}/tool_results"
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json=body)
    except Exception:
        pass

    # --- Resolve which conversation this tool result belongs to ---
    # Note: the tool_request block sets "operation_id": conv_id (see _stream_custom_model),
    # so in practice the client posts here using the conversation_id as the path segment.
    conv_id = str(body.get("conversation_id") or "") or None
    if not conv_id and operation_id in LOCAL_CONVERSATIONS:
        conv_id = operation_id
    if not conv_id:
        for cid, c in LOCAL_CONVERSATIONS.items():
            if c.get("_active_operation_id") == operation_id:
                conv_id = cid
                break

    conv = LOCAL_CONVERSATIONS.get(conv_id) if conv_id else None
    if conv is None:
        # Nothing we can continue - just acknowledge like before
        return JSONResponse(content={"ok": True})

    # --- Normalize incoming tool result(s) into individual tool messages ---
    raw_results = body.get("results") if isinstance(body.get("results"), list) else [body]
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        tc_id = str(entry.get("tool_call_id") or entry.get("id") or entry.get("tool_request_id") or uuid.uuid4().hex)
        output = entry.get("output")
        if output is None:
            output = entry.get("result")
        if output is None:
            output = entry.get("content")
        if output is None:
            output = entry.get("text", "")
        if not isinstance(output, str):
            try:
                output = json.dumps(output)
            except Exception:
                output = str(output)
        conv["messages"].append({"role": "tool", "tool_call_id": tc_id, "content": output})

    save_json(LOCAL_CONVERSATIONS_FILE, LOCAL_CONVERSATIONS)

    # --- Continue the model turn now that it has the tool output ---
    model_id = str(body.get("model") or "freellmapi/auto")
    assistant_seq = conv["next_seq"]
    conv["next_seq"] = assistant_seq + 1
    save_json(LOCAL_CONVERSATIONS_FILE, LOCAL_CONVERSATIONS)

    new_operation_id = uuid.uuid4().hex
    conv["_active_operation_id"] = new_operation_id
    with OPERATION_LOCK:
        OPERATION_EVENTS[new_operation_id] = {"status": "running", "events": [], "seq": 0}

    tool_calls_acc: list = []
    # Run the (blocking) streaming call in a worker thread and wait for it, so that
    # by the time the client re-fetches the timeline, the model's follow-up reply
    # is already saved into conv["messages"].
    await asyncio.to_thread(
        _stream_custom_model,
        new_operation_id, model_id, conv["messages"], conv_id, assistant_seq, tool_calls_acc,
    )

    return JSONResponse(content={"ok": True, "operation_id": new_operation_id})

@app.post("/tools/gateway/call")
async def tools_gateway_call_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    tool_name = str(body.get("tool_name", "") or body.get("name", ""))
    args = body.get("arguments", body.get("args", {}))
    conv_id = str(body.get("conversation_id", body.get("conv_id", "")))
    
    if not tool_name:
        return JSONResponse({"ok": False, "error": "Missing tool_name"}, status_code=400)
        
    result = tools_gateway_call(tool_name, args, conv_id)
    return JSONResponse(content=result)

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

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{BROKER_URL.rstrip('/')}/api/agent/chat", json=body)
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        except Exception as e:
            return JSONResponse({"error": f"Failed to connect to broker: {e}"}, status_code=502)

@app.get("/api/agent/status/{job_id}")
async def proxy_to_broker_status(job_id: str):
    """Polls job status from the broker service."""
    async with httpx.AsyncClient(timeout=30.0) as client:
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

AUTO_EXEC_TOOLS = {"web_search", "run_cmds"}
MAX_AUTO_TOOL_ITERS = 4

def _run_colab_agent_job(message: str, timeout_s: float = 90.0, poll_interval: float = 2.0) -> str:
    """Submits a task to the broker's Colab job queue and blocks until it's completed.
    This is what actually wakes up the Colab worker (it long-polls /api/agent/pending)."""
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(f"{BROKER_URL.rstrip('/')}/api/agent/chat", json={"message": message})
            if r.status_code != 200:
                return f"Error: broker rejected job ({r.status_code}): {r.text[:200]}"
            job_id = r.json().get("job_id")
            if not job_id:
                return "Error: broker did not return a job_id."

            deadline = time.time() + timeout_s
            while time.time() < deadline:
                sr = client.get(f"{BROKER_URL.rstrip('/')}/api/agent/status/{job_id}")
                if sr.status_code == 200:
                    data = sr.json()
                    if data.get("status") == "completed":
                        return str(data.get("response") or "(no output)")
                time.sleep(poll_interval)
            return "Error: timed out waiting for the Colab worker to pick up and finish this job."
    except Exception as e:
        return f"Error contacting broker/Colab worker: {e}"

def _execute_auto_tool(name: str, args: dict) -> str:
    if name == "web_search":
        query = str(args.get("query", ""))
        return _run_colab_agent_job(
            f"Perform a web search for: {query}\nSummarize the most relevant results, including source URLs."
        )
    if name == "run_cmds":
        commands = str(args.get("commands", ""))
        return _run_colab_agent_job(
            f"Run the following command(s) in the Colab VM and report the full output:\n{commands}"
        )
    return f"Unknown auto tool: {name}"

def _stream_custom_model(operation_id: str, model_id: str, messages: list, conv_id: str,
                         assistant_seq: int, tool_calls_acc: list):
    """Background worker: proxies streaming requests via the model broker. Tool calls for
    web_search/run_cmds are auto-executed through the Colab job queue and fed back to the
    model in a loop; any other tool call is surfaced to the client as a permission block."""
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
    current_messages = list(messages)
    try:
        url = f"{BROKER_URL.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8765",
            "X-Title": "AgileBot Gateway"
        }
        if FREELLMAPI_KEY:
            headers["Authorization"] = f"Bearer {FREELLMAPI_KEY}"

        for iteration in range(MAX_AUTO_TOOL_ITERS + 1):
            sys_msgs = list(current_messages)
            if SYSTEM_PROMPT:
                if not sys_msgs or sys_msgs[0].get("role") != "system":
                    sys_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + sys_msgs
                else:
                    sys_msgs[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + sys_msgs[0].get("content", "")}

            tools = _build_tools_payload()
            body = {
                "model": model_id if model_id in CUSTOM_MODELS else "auto",
                "messages": sys_msgs,
                "stream": True
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"

            iter_tool_calls: list = []
            assistant_msg_dict = None

            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"Broker stream error {resp.status_code}: {resp.read().decode()[:200]}")
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
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index", 0)
                                while len(iter_tool_calls) <= idx:
                                    iter_tool_calls.append({"id": "", "name": "", "arguments": ""})
                                if tc.get("id"):
                                    iter_tool_calls[idx]["id"] = tc["id"]
                                if tc.get("function", {}).get("name"):
                                    iter_tool_calls[idx]["name"] = tc["function"]["name"]
                                if tc.get("function", {}).get("arguments"):
                                    iter_tool_calls[idx]["arguments"] += tc["function"]["arguments"]

            # Reconstruct the assistant message for this turn (needed so the model sees its
            # own prior tool_calls when we feed tool results back in the next iteration).
            parsed_calls = []
            for tc in iter_tool_calls:
                if not tc.get("name"):
                    continue
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except Exception:
                    args = {}
                parsed_calls.append({"id": tc.get("id") or uuid.uuid4().hex, "name": tc["name"], "arguments": args})

            if parsed_calls:
                assistant_msg_dict = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": pc["id"],
                            "type": "function",
                            "function": {"name": pc["name"], "arguments": json.dumps(pc["arguments"])},
                        }
                        for pc in parsed_calls
                    ],
                }
                current_messages.append(assistant_msg_dict)

            if not parsed_calls:
                # No tool calls -> model is done, final text already streamed in `acc`.
                break

            auto_calls = [tc for tc in parsed_calls if tc["name"] in AUTO_EXEC_TOOLS]
            client_calls = [tc for tc in parsed_calls if tc["name"] not in AUTO_EXEC_TOOLS]

            if client_calls or iteration >= MAX_AUTO_TOOL_ITERS:
                # Surface remaining calls to the client for execution (e.g. Roblox-side tools),
                # same as before. Any auto tools mixed in here also get surfaced rather than
                # silently dropped, so nothing gets lost.
                for tc in parsed_calls:
                    req_id = tc["id"]
                    tool_request = {
                        "id": req_id, "tool_name": tc["name"], "name": tc["name"],
                        "arguments": tc["arguments"], "args": tc["arguments"],
                        "conversation_id": conv_id, "operation_id": conv_id
                    }
                    _op_emit(operation_id, "block_upsert", {"block": {
                        "render_id": f"tool_request:{req_id}", "id": req_id, "role": "permission",
                        "text": "", "tool_request": tool_request, "tool_request_id": req_id,
                        "operation_id": conv_id, "status": "pending"
                    }})
                    tool_calls_acc.append(tc)
                break

            # All tool calls this turn are auto-executable (web_search/run_cmds) -> run them
            # via the Colab job queue, feed the results back, and loop for a follow-up answer.
            for tc in auto_calls:
                result_text = _execute_auto_tool(tc["name"], tc["arguments"])
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })
                tool_calls_acc.append(tc)
            # loop continues to next iteration for the model's follow-up

        _op_emit(operation_id, "block_patch", {
            "block_id": render_id,
            "patch": {"streaming": False, "text": acc},
        })
        _op_finish(operation_id, "completed")

        conv = LOCAL_CONVERSATIONS.get(conv_id)
        if conv is not None:
            if acc:
                conv["messages"].append({"role": "assistant", "content": acc})
            else:
                # Persist whatever intermediate messages (tool calls/results) were produced,
                # even if this turn ended on a pending client tool request with no text yet.
                conv["messages"] = current_messages
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
