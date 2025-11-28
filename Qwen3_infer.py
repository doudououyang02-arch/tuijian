"""Qwen3-based text retrieval module aligned with the existing recommender API."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import util
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

# Local default to match the provided working setup; override with QWEN_MODEL_PATH if needed.
DEFAULT_QWEN_MODEL = os.getenv(
    "QWEN_MODEL_PATH", "/home/s50052424/UIRecommend/Qwen3-Embedding-8B/"
)

try:
    # Reuse the common result type to minimize UI branching.
    from text_recommand import TextRecommendationResult  # type: ignore
except Exception:  # pragma: no cover - defensive for missing dependency
    @dataclass
    class TextRecommendationResult:  # fallback to keep type stability when imports fail
        source_path: str
        score: Optional[float]
        display_path: Path
        description: Optional[str] = None

        @property
        def filename(self) -> str:
            return Path(self.source_path).name


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last non-masked token as embedding (handles left padding)."""

    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class QwenTextRecommender:
    """Use Qwen3 embeddings for text-to-image recommendation."""

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN_MODEL,
        embeddings_path: Optional[str] = None,
        image_paths_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or DEFAULT_QWEN_MODEL
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.image_paths_path = Path(image_paths_path) if image_paths_path else None
        self.device = device

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._embeddings: Optional[torch.Tensor] = None
        self._image_paths: Optional[List[str]] = None

    @property
    def _model_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        self._model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.bfloat16)
        self._model = self._model.to(self._model_device).eval()

    def _load_embeddings(self) -> torch.Tensor:
        if self._embeddings is None:
            if not self.embeddings_path:
                raise ValueError("Embeddings path is required for text recommendations.")
            if not self.embeddings_path.exists():
                raise FileNotFoundError(f"Text embeddings not found: {self.embeddings_path}")
            # Keep embeddings on the same device as the model for similarity search.
            array = np.load(self.embeddings_path)
            self._embeddings = torch.tensor(array, device=self._model_device)
        return self._embeddings

    def _load_image_paths(self) -> List[str]:
        if self._image_paths is None:
            if not self.image_paths_path:
                raise ValueError("Image paths matrix is required for text recommendations.")
            if not self.image_paths_path.exists():
                raise FileNotFoundError(f"Image path matrix not found: {self.image_paths_path}")
            paths = np.load(self.image_paths_path)
            self._image_paths = [str(p).replace("mnt/vdb2t_1/sujunyan/label30000", "home/s50052424/UIRecommend") for p in paths.tolist()]
        return self._image_paths

    def _encode(self, texts: Iterable[str]) -> torch.Tensor:
        self._load_model()
        assert self._tokenizer is not None and self._model is not None

        batch_dict = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(self._model_device)

        with torch.no_grad():
            outputs = self._model(**batch_dict)
        embeddings = _last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        destination_dir: Optional[Path] = None,
        include_descriptions: bool = False,
    ) -> List[TextRecommendationResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not query:
            raise ValueError("Query must not be empty.")

        embeddings = self._load_embeddings()
        image_paths = self._load_image_paths()
        query_embedding = self._encode([query])[0]

        with torch.no_grad():
            scores = util.cos_sim(query_embedding.float(), embeddings).reshape(-1)
        top_indices = torch.argsort(scores, descending=True)[:top_k]

        copy_destination: Optional[Path] = None
        if destination_dir is not None:
            copy_destination = Path(destination_dir)
            copy_destination.mkdir(parents=True, exist_ok=True)

        results: List[TextRecommendationResult] = []
        for idx in top_indices.tolist():
            source_path = Path(image_paths[idx])
            score = float(scores[idx].item()) if scores is not None else None
            display_path = source_path
            if copy_destination is not None:
                target_file = copy_destination / source_path.name
                if source_path.exists():
                    if source_path.resolve() != target_file.resolve():
                        shutil.copy2(source_path, target_file)
                    display_path = target_file
                else:
                    display_path = target_file

            description = None
            if include_descriptions:
                description = self.get_description(str(source_path))
            results.append(TextRecommendationResult(str(source_path), score, display_path, description))

        return results

    def recommend_from_image(
        self,
        image_path: str,
        top_k: int = 10,
        destination_dir: Optional[Path] = None,
        include_descriptions: bool = False,
    ) -> List[TextRecommendationResult]:
        description = self.get_description(image_path)
        if not description:
            raise FileNotFoundError(
                "Unable to locate description JSON for the selected image; cannot start follow-up search."
            )
        return self.recommend(description, top_k, destination_dir, include_descriptions)

    @staticmethod
    def get_description(image_path: str) -> Optional[str]:
        if not image_path:
            return None
        json_path = Path(image_path).with_suffix(".json")
        candidate = str(json_path)
        if "final_good" in candidate:
            candidate = candidate.replace("final_good", "final_good_json")
        json_path = Path(candidate)
        if not json_path.exists():
            return None
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return json.dumps(data, ensure_ascii=False, indent=2)


__all__ = ["QwenTextRecommender", "DEFAULT_QWEN_MODEL"]
