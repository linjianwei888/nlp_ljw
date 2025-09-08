from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Dict
import os
import yaml

from text_similarity_utils import compute_topk_similarity
from libs.utils.open_embed import OpenEmbeddings


APP_TITLE = "Text Similarity Service"
APP_DESCRIBE = "Compute text similarity using external or local embeddings"
API_PREFIX = "/api/v1"


def create_embedder() -> OpenEmbeddings:
    """Create an embedder from YAML config or environment overrides."""
    # Env overrides
    model_key = os.getenv("EMBED_MODEL_KEY")
    model_name = os.getenv("EMBED_MODEL_NAME")
    server_url = os.getenv("EMBED_SERVER_URL")
    context_length_env = os.getenv("EMBED_CONTEXT_LENGTH")
    context_length = int(context_length_env) if context_length_env else 1024

    disable_remote = os.getenv("EMBED_DISABLE_REMOTE", "0") in {"1", "true", "True"}

    if not disable_remote and not (model_name and server_url):
        # Try load from YAML
        config_path = os.getenv("EMBED_CONFIG_PATH", "/workspace/configs/base_config.yml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            embeddings_cfg = cfg.get("embeddings", {})
            # Choose by key or first
            if model_key and model_key in embeddings_cfg:
                selected = embeddings_cfg[model_key]
            else:
                # pick first available
                selected = next(iter(embeddings_cfg.values())) if embeddings_cfg else None
            if selected:
                model_name = selected.get("name")
                server_url = selected.get("server_url")
                context_length = int(selected.get("context_length", context_length))
        except Exception as e:
            logger.warning(f"Failed to read embed config: {e}")

    # Force local when disabled
    if disable_remote:
        server_url = ""

    # Even if server_url is None/empty, OpenEmbeddings has local fallback
    return OpenEmbeddings(name=model_name or "local-fallback", server_url=server_url or "", context_length=context_length)


app = FastAPI(title=APP_TITLE, description=APP_DESCRIBE)
router = APIRouter(prefix=API_PREFIX)

# Global embedder instance
embedder = create_embedder()


class ComputeTopkSimilarityInput(BaseModel):
    """文本向量相似度TopK输入参数"""

    query_texts: List[str] = Field(default=[], title="查询文本列表")
    candidate_texts: List[str] = Field(default=[], title="候选文本列表")
    top_k: int = Field(default=1, title="返回TopK个最相似")


class ComputeTopkSimilarityOutput(BaseModel):
    """文本向量相似度TopK输出参数"""

    result: Dict = Field(default={}, title="相似度结果")
    errCode: int = Field(default=0, title="状态码")
    errMsg: str = Field(default="success", title="返回信息")
    success: bool = Field(default=True, title="返回状态")


@router.get("/status")
async def get_status():
    return {"status": "running"}


@router.post(
    "/compute_topk_similarity",
    summary="计算文本向量相似度TopK",
    response_model=ComputeTopkSimilarityOutput,
)
async def compute_topk_similarity_api(
    input_data: ComputeTopkSimilarityInput,
) -> ComputeTopkSimilarityOutput:
    response = ComputeTopkSimilarityOutput()
    try:
        query_list = input_data.query_texts
        candidate_list = input_data.candidate_texts
        top_k = input_data.top_k

        if not isinstance(query_list, list) or not isinstance(candidate_list, list):
            raise ValueError("query_texts和candidate_texts必须为列表类型")

        scores, indices = compute_topk_similarity(
            embedder, query_list, candidate_list, top_k=top_k
        )
        response.result = {"scores": scores, "indices": indices}
        response.success = True
    except Exception as e:
        logger.exception(f"compute_topk_similarity接口异常: {e}")
        response.success = False
        response.errCode = 500
        response.errMsg = f"计算相似度异常: {e}"
    return response


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Uvicorn target
# export: app
