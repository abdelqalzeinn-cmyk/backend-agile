# AgileBot Custom Backend

Backend at `C:\Users\abdel\agilebot-backend\` that sits between the AgileBot plugin/site and `api.agilebot.dev`.

## Run
```bash
cd C:\Users\abdel\agilebot-backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

Server runs on `http://localhost:8765`.

## What it does
- **Proxies** `/v1/*` requests to `https://api.agilebot.dev`
- **Adds custom tools** via `POST /admin/tools`
- **Adds custom models** via `POST /admin/models`
- **Tool execution** via `POST /tools/call`
- **Health check** at `/health`

## Plugin integration
Point the plugin's `server_base_url` at `http://localhost:8765` and add:
- Custom tools to `/admin/tools`
- Custom models to `/admin/models`
- The plugin will see them via `/models/custom`

## Environment
- `AGILEBOT_UPSTREAM` — upstream API URL (default: `https://api.agilebot.dev`)
- `PORT` — listen port (default: `8765`)
