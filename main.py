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


class UpstreamUnavailable(Exception):
    def __init__(self, message: str, url: str):
        super().__init__(message)
        self.message = message
        self.url = url


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


# --- Updated routes ---

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


@app.post("/conversations")
async def conversations_post(request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    body = await request.body()
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

    r, err = _safe_proxy("POST", "/conversations", body=body, headers=h)
    if err:
        return JSONResponse(content=err, status_code=502)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))


@app.post("/conversations/{conversation_id}/messages")
async def conversations_messages_post(conversation_id: str, request: Request):
    h = dict(_proxy_headers())
    a = request.headers.get("authorization") or request.headers.get("Authorization")
    if a:
        h["authorization"] = a
    body = await request.body()
    try:
        body_json = json.loads(body) if body else {}
    except Exception:
        body_json = {}
    model_id = str(body_json.get("model", ""))

    if _is_custom_model(model_id) or conversation_id in LOCAL_CONVERSATIONS:
        try:
            result = _handle_custom_conversation(model_id, body_json.get("message", ""), conversation_id)
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            return JSONResponse({"detail": {"code": "custom_model_error", "model": model_id, "message": str(e)}}, status_code=502)

    r, err = _safe_proxy("POST", f"/conversations/{conversation_id}/messages", body=body, headers=h)
    if err:
        return JSONResponse(content=err, status_code=502)
    return Response(content=r.content, status_code=r.status_code, headers=_clean_response_headers(r.headers))
