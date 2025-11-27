"""Qwen-based text retrieval helper aligned with the existing demo API."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

default_qwen_model = "Qwen/Qwen2.5-7B-Instruct"

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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class QwenTextRecommender:
    """Use a Qwen model to embed text for image retrieval.

    The interface mirrors ``TextRecommender`` so the Flask app can swap models
    with minimal branching. Heavy dependencies are imported lazily so the file
    can be compiled without the runtime model being present.
    """

    def __init__(
        self,
        model_name: str = default_qwen_model,
        embeddings_path: Optional[str] = None,
        image_paths_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or default_qwen_model
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.image_paths_path = Path(image_paths_path) if image_paths_path else None
        self.device = device

        self._tokenizer = None
        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._image_paths: Optional[List[str]] = None

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        if self.device:
            self._model = self._model.to(self.device)
        else:
            # Fall back to GPU when available without forcing the dependency.
            if torch.cuda.is_available():
                self._model = self._model.cuda()

    def _load_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            if not self.embeddings_path:
                raise ValueError("Embeddings path is required for text recommendations.")
            if not self.embeddings_path.exists():
                raise FileNotFoundError(f"Text embeddings not found: {self.embeddings_path}")
            self._embeddings = np.load(self.embeddings_path)
        return self._embeddings

    def _load_image_paths(self) -> List[str]:
        if self._image_paths is None:
            if not self.image_paths_path:
                raise ValueError("Image paths matrix is required for text recommendations.")
            if not self.image_paths_path.exists():
                raise FileNotFoundError(f"Image path matrix not found: {self.image_paths_path}")
            paths = np.load(self.image_paths_path)
            self._image_paths = [str(p) for p in paths.tolist()]
        return self._image_paths

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        self._load_model()
        assert self._tokenizer is not None and self._model is not None
        import torch

        encoded = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        if self.device:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        elif next(self._model.parameters()).is_cuda:
            encoded = {k: v.cuda() for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy()

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
        scores = _cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        copy_destination: Optional[Path] = None
        if destination_dir is not None:
            copy_destination = Path(destination_dir)
            copy_destination.mkdir(parents=True, exist_ok=True)

        results: List[TextRecommendationResult] = []
        for idx in top_indices:
            source_path = Path(image_paths[idx])
            score = float(scores[idx]) if scores is not None else None
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


__all__ = ["QwenTextRecommender", "default_qwen_model"]
