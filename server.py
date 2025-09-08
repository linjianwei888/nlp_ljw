from fastapi import FastAPI, APIRouter, UploadFile, File, Form
from loguru import logger
from typing import List, Dict

from libs.utils.text_similarity_utils import compute_topk_similarity
from application.bidder_location.bidder_loaction import BidderLocation
from application.bidder_location.multi_file_bidder_location_v3 import (
    MultiFileBidderLocation,
)
from libs.schema.bidder_location import (
    BidderLocationInput,
    BidderLocationOutput,
    ComputeTopkSimilarityOutput,
    ComputeTopkSimilarityInput,
)
from libs.utils.common import urljoin
from libs.utils.constant import (
    APP_VERSION,
    APP_TITLE,
    APP_DESCRIBE,
    KAFKA_PATH,
    base_config,
)

from libs.utils.web_utils import set_middleware

_APP = FastAPI(title=APP_TITLE, description=APP_DESCRIBE)
prefix_route = urljoin("/bidder_parse/", APP_VERSION)
router = APIRouter(prefix=prefix_route)

bidder_location = BidderLocation()
multi_file_bidder_location = MultiFileBidderLocation()
embedder = multi_file_bidder_location.embedder

class ComputeTopkSimilarityInput(BaseModel):
    """文本向量相似度TopK输入参数"""

    query_texts: List[str] = Field(default=[], title="查询文本列表")
    candidate_texts: List[str] = Field(default=[], title="候选文本列表")
    top_k: int = Field(default=1, title="返回TopK个最相似")

    class Config:
        extra = Extra.allow


class ComputeTopkSimilarityOutput(BaseModel):
    """文本向量相似度TopK输出参数（参考BidderLocationOutput）"""

    result: Dict = Field(default={}, title="相似度结果")

    errCode: int = Field(default=0, title="状态码")
    errMsg: str = Field(default="success", title="返回信息")
    success: bool = Field(default=True, title="返回状态")


@router.get("/status")
async def get_status():
    """健康检查状态"""
    return {"status": "running"}



@router.post(
    "/compute_topk_similarity",
    summary="计算文本向量相似度TopK",
    response_model=ComputeTopkSimilarityOutput,
)
async def compute_topk_similarity_api(
    input_data: ComputeTopkSimilarityInput,
) -> ComputeTopkSimilarityOutput:
    """
    计算两组文本的向量余弦相似度，并返回每个query_texts与candidate_texts最相似的top_k分数和索引。
    """

    response = ComputeTopkSimilarityOutput()
    try:
        query_list = input_data.query_texts
        candidate_list = input_data.candidate_texts
        top_k = input_data.top_k

        # 校验输入
        if not isinstance(query_list, list) or not isinstance(candidate_list, list):
            raise ValueError("query_texts和candidate_texts必须为列表类型")

        # 获取embedder实例（假设multi_file_bidder_location有embedder属性）
        scores, indices = compute_topk_similarity(
            embedder, query_list, candidate_list, top_k=top_k
        )
        response.result = {"scores": scores, "indices": indices}
        response.success = True
        response.errCode = 0
        response.errMsg = "success"
    except Exception as e:
        logger.exception(f"compute_topk_similarity接口异常: {e}")
        response.success = False
        response.errCode = 500
        response.errMsg = f"计算相似度异常: {e}"
        response.locationResult = {}
    return response


# 设置中间件
set_middleware(_APP, "*")

_APP.include_router(router)
