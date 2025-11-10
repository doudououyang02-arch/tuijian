"""Image-based recommendation utilities for the web demo.

This module wraps the original prototype script into a reusable service
object that can be imported by the Flask application.  The actual model
and embedding files live on the target deployment machine, so the helper
class defers loading heavy assets until they are required.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

# Default locations match the original standalone script.  They can be
# overridden at runtime through the ImageRecommender constructor.
DEFAULT_MODEL_PATH = "./siglip2"
DEFAULT_EMBEDDINGS_PATH = (
    "/mnt/vdb2t_1/sujunyan/program/ui/image_features_extract/embedding/clean_embeddings.npy"
)
DEFAULT_IMAGE_PATH_MATRIX = (
    "/mnt/vdb2t_1/sujunyan/program/ui/image_features_extract/embedding/clean_paths.npy"
)


@dataclass
class RecommendationResult:
    """Container describing a single recommended image."""

    source_path: str
    score: Optional[float]
    display_path: Path

    @property
    def filename(self) -> str:
        return Path(self.source_path).name


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cosine_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all rows of two matrices."""

    if matrix_a.ndim == 1:
        matrix_a = matrix_a.reshape(1, -1)
    if matrix_b.ndim == 1:
        matrix_b = matrix_b.reshape(1, -1)

    a_norm = matrix_a / (np.linalg.norm(matrix_a, axis=1, keepdims=True) + 1e-12)
    b_norm = matrix_b / (np.linalg.norm(matrix_b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class ImageRecommender:
    """High level helper for generating image-to-image recommendations."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
        image_paths_path: str = DEFAULT_IMAGE_PATH_MATRIX,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.image_paths_path = Path(image_paths_path) if image_paths_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._processor = None
        self._model = None
        self._embeddings = None
        self._image_paths: Optional[List[str]] = None

    # Heavy dependencies are instantiated lazily so that importing the
    # module does not immediately require GPU resources.
    def _load_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        from transformers import AutoImageProcessor, SiglipVisionModel

        if not self.model_path:
            raise ValueError("Model path must be provided for image recommendations.")

        self._processor = AutoImageProcessor.from_pretrained(self.model_path)
        self._model = SiglipVisionModel.from_pretrained(self.model_path).to(self.device).eval()

    def _load_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            if not self.embeddings_path:
                raise ValueError("Embeddings path is required for image recommendations.")
            if not self.embeddings_path.exists():
                raise FileNotFoundError(f"Embedding file not found: {self.embeddings_path}")
            self._embeddings = np.load(self.embeddings_path)
        return self._embeddings

    def _load_image_paths(self) -> List[str]:
        if self._image_paths is None:
            if not self.image_paths_path:
                raise ValueError("Image paths matrix is required for image recommendations.")
            if not self.image_paths_path.exists():
                raise FileNotFoundError(f"Image path matrix not found: {self.image_paths_path}")
            paths = np.load(self.image_paths_path)
            self._image_paths = [str(p) for p in paths.tolist()]
        return self._image_paths

    def _extract_embedding(self, image_path: Path) -> np.ndarray:
        self._load_model()
        assert self._processor is not None and self._model is not None

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            embedding = outputs.pooler_output
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.cpu().numpy()

    def recommend(
        self,
        target_image_path: str,
        top_k: int = 10,
        destination_dir: Optional[Path] = None,
    ) -> List[RecommendationResult]:
        """Return the top-k similar images for ``target_image_path``.

        The recommended images are optionally copied into ``destination_dir``
        so that the web frontend can serve them directly.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        target_path = Path(target_image_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Target image not found: {target_image_path}")

        embeddings_matrix = self._load_embeddings()
        image_paths = self._load_image_paths()
        target_embedding = self._extract_embedding(target_path)

        similarities = _cosine_similarity(target_embedding, embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[RecommendationResult] = []
        copy_destination: Optional[Path] = None
        if destination_dir is not None:
            copy_destination = _ensure_directory(destination_dir)

        for index in top_indices:
            source_path = Path(image_paths[index])
            score = float(similarities[index]) if similarities is not None else None
            display_path = source_path
            if copy_destination is not None:
                target_file = copy_destination / source_path.name
                if source_path.exists():
                    if source_path.resolve() != target_file.resolve():
                        shutil.copy2(source_path, target_file)
                    display_path = target_file
                else:
                    # Fallback: still expose the intended path even if the
                    # original asset is not available on the current machine.
                    display_path = target_file
            results.append(RecommendationResult(str(source_path), score, display_path))

        return results


__all__ = ["ImageRecommender", "RecommendationResult"]
