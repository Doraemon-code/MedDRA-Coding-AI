from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from .utils import dump_json
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())


@dataclass
class IndexMetadata:
    model_name: str
    normalize: bool
    dimension: int
    language: str
    version: str
    terms_count: int
    vector_store: str = "faiss"

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "normalize": self.normalize,
            "dimension": self.dimension,
            "language": self.language,
            "version": self.version,
            "terms_count": self.terms_count,
            "vector_store": self.vector_store,
        }


class IndexBuilder:
    """Vectorise MedDRA terms and persist them to a FAISS index."""

    def __init__(self, model_name: str, *, batch_size: int = 64, normalize: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device="cuda")
        return self._model

    def build(
        self,
        *,
        documents: pd.DataFrame,
        output_dir: Path,
        language: str,
        version: str,
    ) -> IndexMetadata:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        documents = documents.copy()
        documents = documents.reset_index(drop=True)
        documents.to_csv(output_dir / "documents.csv", index=False)

        if "document_text" not in documents.columns:
            raise ValueError("documents DataFrame must contain a 'document_text' column")

        texts = documents["document_text"].tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]
        if self.normalize:
            cpu_index = faiss.IndexFlatIP(dimension)
        else:
            cpu_index = faiss.IndexFlatL2(dimension)
        
        # --- GPU 加速部分（新增） ---
        res = faiss.StandardGpuResources()  # 创建 GPU 资源
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 把 CPU index 转成 GPU index
        
        gpu_index.add(embeddings)  # 在 GPU 上 add
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)  # 转回 CPU index 以便保存
        # --- GPU 加速结束 ---
        
        faiss.write_index(cpu_index, str(output_dir / "index.faiss"))


        # index.add(embeddings)
        # faiss.write_index(index, str(output_dir / "index.faiss"))

        metadata = IndexMetadata(
            model_name=self.model_name,
            normalize=self.normalize,
            dimension=dimension,
            language=language,
            version=version,
            terms_count=len(documents),
            vector_store="faiss",
        )
        dump_json(output_dir / "metadata.json", metadata.to_dict())
        return metadata
