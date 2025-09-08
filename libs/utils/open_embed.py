import os
import hashlib
from typing import List, Any, Union

import numpy as np
import requests
import aiohttp
from aiohttp import ClientTimeout
from loguru import logger


class OpenEmbeddings:
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

        # Local fallback dimension
        self.local_dim = int(os.getenv("LOCAL_EMBED_DIM", "384"))

    def _local_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Deterministic hashing-based embedding as a local fallback.

        Uses simple token hashing into a fixed-size bag-of-hashes vector.
        """
        dim = self.local_dim
        embeddings: List[List[float]] = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            # Basic tokenization by whitespace; include whole text as token fallback
            tokens = (text or "").lower().split()
            if not tokens:
                tokens = [text or ""]
            for token in tokens:
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                vec[idx] += 1.0
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Call out to local embedding endpoint for embedding search docs.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        payload = {
            "input": [text[:self.context_length] for text in texts],
            "model": self.name,
            "encoding_format": "float",
        }
        try:
            if not self.server_url:
                raise RuntimeError("server_url is empty, using local fallback")
            response = requests.post(url=self.server_url, json=payload, timeout=3)
            response.raise_for_status()
            data = response.json()
            return [d["embedding"] for d in data["data"]]
        except Exception as e:
            logger.warning(f"embed_documents fallback to local due to: {e}")
            return self._local_embed_texts(texts)

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Call out to local embedding endpoint for embedding query text.
        Args:
            text: The text to embed.
        Returns:
            Embedding for the text.
        """
        try:
            if not self.server_url:
                raise RuntimeError("server_url is empty, using local fallback")
            response = requests.post(
                url=self.server_url,
                json={"input": text[:self.context_length], "model": self.name, "encoding_format": "float"},
                timeout=3,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"embed_query fallback to local due to: {e}")
            return self._local_embed_texts([text])[0]

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
        if not self.server_url:
            return self._local_embed_texts(texts)
        try:
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.post(
                    url=self.server_url,
                    json={
                        "input": [text[:self.context_length] for text in texts],
                        "model": self.name,
                        "encoding_format": "float",
                    },
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return [d["embedding"] for d in data["data"]]
        except Exception as e:
            logger.warning(f"aembed_documents fallback to local due to: {e}")
            return self._local_embed_texts(texts)

    async def aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        """异步版本查询嵌入"""
        if not self.server_url:
            return self._local_embed_texts([text])[0]
        try:
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.post(
                    url=self.server_url,
                    json={
                        "input": text[:self.context_length],
                        "model": self.name,
                        "encoding_format": "float",
                    },
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"aembed_query fallback to local due to: {e}")
            return self._local_embed_texts([text])[0]

    async def aembed_images(self, images: Union[str, List[str]] = None) -> List[float]:
        """异步版本图片嵌入"""
        if not self.server_url:
            # Not supported locally; return zero vector
            return [0.0] * self.local_dim
        try:
            async with aiohttp.ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.post(
                    url=self.server_url,
                    json={"input": images, "model": self.name},
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("embedding")
        except Exception as e:
            logger.warning(f"aembed_images failed: {e}")
            return [0.0] * self.local_dim
