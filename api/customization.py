# api/customization.py
from fastapi import APIRouter, HTTPException
from schemas import CustomizationRequest, CustomizationResponse

router = APIRouter(prefix="/custom", tags=["Customization"])

@router.post("/apply", response_model=CustomizationResponse, summary="커스텀 메이크업 적용")
async def apply_customization(request: CustomizationRequest):
    """
    얼굴 이미지에 메이크업 intensity를 조정해 적용하는 API
    """
    try:
        from service.customization_service import run_inference
        result = run_inference(request.dict())

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("message", "Inference failed"))

        return CustomizationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")