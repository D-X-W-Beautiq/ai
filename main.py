# main.py
import os
from fastapi import FastAPI
from api.router import api_router
from api.health import router as health_router

ROOT_PATH = os.getenv("ROOT_PATH", "")  # 예: "/proxy/8000" (환경에 맞게)

app = FastAPI(
    title="Beautiq API",
    version="0.1.0",
    root_path=ROOT_PATH,        # ✅ 중요: 프록시 경로
    docs_url="/docs",           # 기본 유지 (root_path가 앞에 자동으로 붙습니다)
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.include_router(health_router)
app.include_router(api_router)  # api_router 내부 prefix가 /v1이면 그대로 유지

@app.get("/", include_in_schema=False)
def _root():
    return {"message": "See docs", "docs": f"{ROOT_PATH}/docs", "openapi": f"{ROOT_PATH}/openapi.json"}
