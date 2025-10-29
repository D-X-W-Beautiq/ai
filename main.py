# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import api_router

app = FastAPI(
    title="Beautiq AI-BE",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (필요 시 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 운영 시 FE/BE 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /v1 아래로 라우터 마운트
app.include_router(api_router, prefix="/v1")


# 선택: 기동 로그/프리로드 훅
@app.on_event("startup")
async def on_startup():
    print("[startup] Beautiq AI-BE is starting ...")


@app.get("/", tags=["meta"])
async def root():
    return {"service": "Beautiq AI-BE", "version": app.version}
