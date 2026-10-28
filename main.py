from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.pipeline.mcap_engine import load_client_configs, run_client_delivery_pipeline
from src.pipeline.models import FaultInjectionConfig

BASE_DIR = Path(__file__).resolve().parent
DELIVERIES_DIR = BASE_DIR / "deliveries"


def execute_pipeline_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = run_client_delivery_pipeline(
        client_id=str(payload.get("client_id", "frontier_manipulation_lab")),
        num_frames=int(payload.get("num_frames", 120)),
        faults=FaultInjectionConfig(
            hand_occlusion_burst=bool(payload.get("hand_occlusion_burst", False)),
            sensor_clock_drift=bool(payload.get("sensor_clock_drift", False)),
            nan_joint_spike=bool(payload.get("nan_joint_spike", False)),
        ),
        override_hand_confidence=(
            float(payload["override_hand_confidence"])
            if payload.get("override_hand_confidence") is not None
            else None
        ),
        override_sync_drift_ms=(
            float(payload["override_sync_drift_ms"])
            if payload.get("override_sync_drift_ms") is not None
            else None
        ),
        output_dir=DELIVERIES_DIR,
    )
    return result.model_dump()


# Optional FastAPI ASGI app when fastapi is installed
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(
        title="Robotics Multimodal MCAP Delivery Pipeline",
        description="Config-driven per-client schema transformation, VLM hand-segmentation QC, and MCAP packaging engine.",
        version="1.0.0",
    )

    @app.get("/", response_class=HTMLResponse)
    def serve_console() -> str:
        return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/clients")
    def list_clients() -> dict:
        clients = load_client_configs()
        return {"clients": {cid: cfg.model_dump() for cid, cfg in clients.items()}}

    @app.post("/api/pipeline/run")
    def execute_pipeline_api(req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return execute_pipeline_payload(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/deliveries/{filename}")
    def download_mcap(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        target = DELIVERIES_DIR / safe_name
        if not target.is_file():
            raise HTTPException(status_code=404, detail="MCAP container file not found.")
        return FileResponse(path=target, media_type="application/octet-stream", filename=safe_name)

except ImportError:
    app = None  # type: ignore


class ConsoleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = (BASE_DIR / "static" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/api/clients":
            clients = load_client_configs()
            body = json.dumps({"clients": {cid: cfg.model_dump() for cid, cfg in clients.items()}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/deliveries/"):
            fname = Path(self.path.split("/api/deliveries/", 1)[1]).name
            target = DELIVERIES_DIR / fname
            if target.is_file():
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/pipeline/run":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            res_dict = execute_pipeline_payload(payload)
            body = json.dumps(res_dict).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "Not Found")


if __name__ == "__main__":
    port = 8095
    server = HTTPServer(("127.0.0.1", port), ConsoleHTTPRequestHandler)
    print(f"Robotics Multimodal MCAP Delivery Console running at http://localhost:{port}")
    server.serve_forever()
