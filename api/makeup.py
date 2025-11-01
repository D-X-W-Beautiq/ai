# api/makeup.py
from fastapi import APIRouter, HTTPException
from schemas import MakeupRequest, MakeupResponse
from typing import Dict, Any

router = APIRouter(prefix="/makeup", tags=["Makeup"])

def _raise_on_failed(service_result: Dict[str, Any]) -> None:
    msg = (service_result or {}).get("message", "Unknown error")
    low = msg.lower()
    if any(k in low for k in ["missing", "invalid base64", "invalid", "not found", "bad"]):
        raise HTTPException(status_code=422, detail=msg)
    raise HTTPException(status_code=500, detail=msg)

@router.post("/simulate", response_model=MakeupResponse, summary="메이크업 시뮬레이션")
def makeup_simulate(req: MakeupRequest):
    try:
        # ⬇️ 지연 임포트
        from service.makeup_service import run_inference as makeup_inference
        service_result = makeup_inference(req.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not isinstance(service_result, dict) or "status" not in service_result:
        raise HTTPException(status_code=500, detail="Invalid service response")
    if service_result.get("status") != "success":
        _raise_on_failed(service_result)
    return {"result": service_result}