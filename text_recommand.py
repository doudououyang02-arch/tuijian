"""Text-based image recommendation utilities for the web demo."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Default file locations taken from the prototype script.
DEFAULT_MODEL_PATH = (
    "/mnt/vdb2t_1/sujunyan/program/ui/image_features_extract/models/baai_bge-m3/baai_bg3-m3"
)
DEFAULT_TEXT_EMBEDDINGS = "text_embeddings_8820.npy"
DEFAULT_PATH_MATRIX = "path_matrix_8820.npy"

# Round-aware defaults to support progressive recall strategies.
DEFAULT_TEXT_EMBEDDINGS_ROUND1 = DEFAULT_TEXT_EMBEDDINGS
DEFAULT_PATH_MATRIX_ROUND1 = DEFAULT_PATH_MATRIX

DEFAULT_TEXT_EMBEDDINGS_ROUND2 = "text_embeddings_8820_all_content.npy"
DEFAULT_PATH_MATRIX_ROUND2 = "path_matrix_8820_content_all_content.npy"

DEFAULT_TEXT_EMBEDDINGS_ROUND3 = "text_embeddings_8820_top_sementic.npy"
DEFAULT_PATH_MATRIX_ROUND3 = "path_matrix_8820_top_sementic.npy"


@dataclass
class TextRecommendationResult:
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


class TextRecommender:
    """Helper that turns text queries into image recommendations."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        embeddings_path: str = DEFAULT_TEXT_EMBEDDINGS,
        image_paths_path: str = DEFAULT_PATH_MATRIX,
        use_fp16: bool = False,
    ) -> None:
        self.model_path = model_path
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.image_paths_path = Path(image_paths_path) if image_paths_path else None
        self.use_fp16 = use_fp16

        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._image_paths: Optional[List[str]] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel

        if not self.model_path:
            raise ValueError("Model path must be provided for text recommendations.")

        self._model = BGEM3FlagModel(self.model_path, use_fp16=self.use_fp16)

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
        assert self._model is not None
        outputs = self._model.encode(
            list(texts), return_dense=True, return_sparse=True, return_colbert_vecs=True
        )
        dense_vectors = outputs["dense_vecs"]
        if isinstance(dense_vectors, list):
            return np.vstack([np.array(vec) for vec in dense_vectors])
        return np.array(dense_vectors)

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

    def get_description(self, image_path: str) -> Optional[str]:
        json_path = self._infer_json_path(image_path)
        if json_path is None or not json_path.exists():
            return None
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return json.dumps(data, ensure_ascii=False, indent=2)

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
    def _infer_json_path(image_path: str) -> Optional[Path]:
        if not image_path:
            return None
        source = Path(image_path)
        json_candidate = source.with_suffix(".json")
        candidate_str = str(json_candidate)
        if "final_good" in candidate_str:
            candidate_str = candidate_str.replace("final_good", "final_good_json")
        return Path(candidate_str)


__all__ = ["TextRecommender", "TextRecommendationResult"]
