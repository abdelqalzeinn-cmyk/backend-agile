import asyncio

# Task queue for Colab workers
PENDING_JOBS: dict = {}
COMPLETED_JOBS: dict = {}

@app.post("/api/agent/chat")
async def user_chat_gateway(request: Request):
    """User sends requests here. Render holds the request while Colab executes tools."""
    body = await request.json()
    user_msg = body.get("message", "").strip()
    if not user_msg:
        return JSONResponse({"error": "Field 'message' is required"}, status_code=400)

    job_id = uuid.uuid4().hex
    job_event = asyncio.Event()
    
    PENDING_JOBS[job_id] = {
        "id": job_id,
        "message": user_msg,
        "event": job_event
    }

    # Wait up to 120 seconds for Colab worker to execute tools & finish
    try:
        await asyncio.wait_for(job_event.wait(), timeout=120.0)
        result = COMPLETED_JOBS.pop(job_id, {"response": "No response returned from worker"})
        return JSONResponse(result)
    except asyncio.TimeoutError:
        PENDING_JOBS.pop(job_id, None)
        return JSONResponse({"error": "Colab worker timed out or is offline"}, status_code=504)

@app.get("/api/agent/pending")
def get_pending_job():
    """Colab worker polls this endpoint to claim pending jobs."""
    for job_id, job in list(PENDING_JOBS.items()):
        if not job.get("assigned"):
            job["assigned"] = True
            return {"job_id": job_id, "message": job["message"]}
    return {"job_id": None}

@app.post("/api/agent/complete/{job_id}")
async def complete_job(job_id: str, request: Request):
    """Colab worker posts execution results back here to respond to the user."""
    body = await request.json()
    if job_id in PENDING_JOBS:
        job = PENDING_JOBS.pop(job_id)
        COMPLETED_JOBS[job_id] = body
        job["event"].set()
        return {"ok": True}
    return JSONResponse({"ok": False, "error": "Job expired or missing"}, status_code=404)
