from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag.chain import get_provider, set_provider

router = APIRouter(prefix="/settings", tags=["settings"])


class ProviderRequest(BaseModel):
    provider: str  # "gemini" or "ollama"


class ProviderResponse(BaseModel):
    provider: str


@router.get("/provider", response_model=ProviderResponse)
def get_current_provider():
    return ProviderResponse(provider=get_provider())


@router.put("/provider", response_model=ProviderResponse)
def switch_provider(request: ProviderRequest):
    if request.provider not in ("gemini", "ollama"):
        raise HTTPException(
            status_code=400,
            detail="Provider must be 'gemini' or 'ollama'"
        )
    set_provider(request.provider)
    return ProviderResponse(provider=request.provider)
