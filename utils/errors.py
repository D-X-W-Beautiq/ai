# -*- coding: utf-8 -*-
from fastapi import HTTPException

class AppError(Exception):
    def __init__(self, message: str, code: str = "APP_ERROR", status: int = 500):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status

def bad_request(msg: str, code: str = "BAD_REQUEST") -> HTTPException:
    return HTTPException(status_code=400, detail={"message": msg, "error_code": code})

def unprocessable(msg: str, code: str = "UNPROCESSABLE_ENTITY") -> HTTPException:
    return HTTPException(status_code=422, detail={"message": msg, "error_code": code})

def internal_error(msg: str, code: str = "INTERNAL_ERROR") -> HTTPException:
    return HTTPException(status_code=500, detail={"message": msg, "error_code": code})

def map_service_error(result: dict) -> HTTPException:
    """service가 {'status':'failed','message':...} 형태로 반환할 때 HTTPException으로 변환"""
    msg = (result or {}).get("message", "Unknown error")
    low = msg.lower()
    if any(k in low for k in ["missing", "invalid", "bad", "not found", "base64", "json"]):
        return unprocessable(msg)
    return internal_error(msg)
