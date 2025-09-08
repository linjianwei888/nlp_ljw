import requests
import aiohttp
from typing import List, Any, Union

from loguru import logger
from aiohttp import ClientTimeout
from langchain.embeddings.base import Embeddings


class OpenEmbeddings(Embeddings):
    def __init__(
            self,
            name: str,
            server_url: str,
            context_length: int = 1024,
    ):
        self.name = name
        self.server_url = server_url

        if context_length:
            self.context_length = context_length - 2  # minus two for [cls] and [sep]/[eos]
        else:
            self.context_length = context_length

    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Call out to local embedding endpoint for embedding search docs.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        response = requests.post(
            url=self.server_url,
            json={"input": [text[:self.context_length] for text in texts], "model": self.name, "encoding_format": "float"}
        )
        data = response.json()
        logger.info(f"批量embeddings成功")

        return [d["embedding"] for d in data["data"]]

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Call out to local embedding endpoint for embedding query text.
        Args:
            text: The text to embed.
        Returns:
            Embedding for the text.
        """
        response = requests.post(
            url=self.server_url, json={"input": text[:self.context_length], "model": self.name, "encoding_format": "float"}
        )
        data = response.json()

        return data["data"][0]["embedding"]

    def embed_images(self, images: Union[str, List[str]] = None) -> List[float]:
        """Call out to local embedding endpoint for embedding query text.
        Args:
            images: The images to embed.
        Returns:
            Embedding for the images.
        """
        response = requests.post(
            url=self.server_url, json={"input": images, "model": self.name}
        )
        data = response.json()

        return data["embedding"]

    async def aembed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """异步版本文档嵌入"""
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                url=self.server_url,
                json={
                    "input": [text[:self.context_length] for text in texts],
                    "model": self.name,
                    "encoding_format": "float"
                }
            )
            data = await response.json()
            return [d["embedding"] for d in data["data"]]

    async def aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        """异步版本查询嵌入"""
        async with aiohttp.ClientSession(timeout=ClientTimeout(total=60 * 60)) as session:
            response = await session.post(
                url=self.server_url,
                json={
                    "input": text[:self.context_length],
                    "model": self.name,
                    "encoding_format": "float"
                }
            )
            data = await response.json()
            return data["data"][0]["embedding"]

    async def aembed_images(self, images: Union[str, List[str]] = None) -> List[float]:
        """异步版本图片嵌入"""
        async with aiohttp.ClientSession(timeout=ClientTimeout(total=60 * 60)) as session:
            response = await session.post(
                url=self.server_url,
                json={"input": images, "model": self.name}
            )
            data = await response.json()
            return data["embedding"]
