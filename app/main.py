from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routers import api_router
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Gauge, start_http_server
from ai.model_loader import build_and_load_model
import psutil
import threading
import time
import os

# 라벨러: path를 실제 요청 URL로 그대로 쓰도록 설정
def custom_handler_label(info: Info) -> str:
    return info.request.url.path  # ← 이게 핵심!

instrumentator = Instrumentator(
    should_instrument_requests_inprogress=True,
    excluded_handlers=[],
    should_respect_env_var=False
)

# 📌 Lifespan 정의
@asynccontextmanager
async def lifespan(api: FastAPI):
    model_path = "ai/model/handwriting_hybrid_model_1.keras"

    if not os.path.exists(model_path):
        raise RuntimeError(f"모델 경로가 존재하지 않습니다: {model_path}")

    api.state.model = build_and_load_model(model_path)
    print("✅ 모델이 앱에 로드되었습니다.")
    yield
    print("🔚 FastAPI 종료 중...")

app = FastAPI(
    title="끄적 프로젝트 AI 서버",
    description="API Documentation",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

instrumentator.instrument(app).expose(app)

cpu_usage_gauge = Gauge("app_cpu_usage_percent", "CPU usage percent")
thread_count_gauge = Gauge("app_thread_count", "Number of threads")
memory_usage_gauge = Gauge("app_memory_usage_bytes", "Memory usage in bytes")

# 별도 스레드로 주기적으로 수집
def system_metrics_collector():
    process = psutil.Process(os.getpid())  # ✅ 현재 프로세스
    while True:
        cpu_usage_gauge.set(psutil.cpu_percent())
        thread_count_gauge.set(threading.active_count())
        memory_usage_gauge.set(process.memory_info().rss)
        time.sleep(5)

# 수집기 시작
start_http_server(8001)  # 시스템 메트릭은 별도 포트
threading.Thread(target=system_metrics_collector, daemon=True).start()

