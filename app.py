from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, flash, render_template, request, url_for

from image_recommend import (
    DEFAULT_EMBEDDINGS_PATH as DEFAULT_IMAGE_EMBEDDINGS,
    DEFAULT_IMAGE_PATH_MATRIX,
    DEFAULT_MODEL_PATH as DEFAULT_IMAGE_MODEL,
    ImageRecommender,
    RecommendationResult,
)
from text_recommand import (
    DEFAULT_MODEL_PATH as DEFAULT_TEXT_MODEL,
    DEFAULT_PATH_MATRIX,
    DEFAULT_TEXT_EMBEDDINGS,
    TextRecommendationResult,
    TextRecommender,
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "image-recommend-demo")

BASE_DIR = Path(__file__).resolve().parent
STATIC_ROOT = BASE_DIR / "static"
UPLOAD_DIR = STATIC_ROOT / "uploads"
RESULTS_DIR = STATIC_ROOT / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Configuration values can be overridden with environment variables at deploy
# time to match the target machine's folder layout.
image_recommender: Optional[ImageRecommender] = None
text_recommender: Optional[TextRecommender] = None


def _create_image_service() -> ImageRecommender:
    global image_recommender
    if image_recommender is None:
        model_path = os.getenv("IMAGE_MODEL_PATH")
        embedding_path = os.getenv("IMAGE_EMBEDDINGS_PATH")
        image_paths_path = os.getenv("IMAGE_PATHS_MATRIX")
        image_recommender = ImageRecommender(
            model_path=model_path or DEFAULT_IMAGE_MODEL,
            embeddings_path=embedding_path or DEFAULT_IMAGE_EMBEDDINGS,
            image_paths_path=image_paths_path or DEFAULT_IMAGE_PATH_MATRIX,
        )
    return image_recommender


def _create_text_service() -> TextRecommender:
    global text_recommender
    if text_recommender is None:
        model_path = os.getenv("TEXT_MODEL_PATH")
        embeddings_path = os.getenv("TEXT_EMBEDDINGS_PATH")
        paths_path = os.getenv("TEXT_PATHS_MATRIX")
        text_recommender = TextRecommender(
            model_path=model_path or DEFAULT_TEXT_MODEL,
            embeddings_path=embeddings_path or DEFAULT_TEXT_EMBEDDINGS,
            image_paths_path=paths_path or DEFAULT_PATH_MATRIX,
        )
    return text_recommender


def _build_result_payload(
    results: List[RecommendationResult],
    debug_enabled: bool,
    include_descriptions: bool = False,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    static_root = STATIC_ROOT.resolve()
    text_service: Optional[TextRecommender] = None

    for item in results:
        display_path = Path(item.display_path)
        if display_path.exists():
            try:
                relative = display_path.resolve().relative_to(static_root)
                image_url = url_for("static", filename=str(relative).replace(os.sep, "/"))
            except ValueError:
                image_url = display_path.as_posix()
        else:
            image_url = display_path.as_posix()

        description: Optional[str] = None
        if include_descriptions:
            if isinstance(item, TextRecommendationResult):
                description = item.description
            else:
                if text_service is None:
                    text_service = _create_text_service()
                description = text_service.get_description(item.source_path)

        payload.append(
            {
                "image_url": image_url,
                "source_path": item.source_path,
                "score": None if item.score is None else f"{item.score:.4f}",
                "description": description,
                "filename": item.filename,
            }
        )
    return payload


def _save_upload(file_storage) -> Path:
    upload_id = uuid.uuid4().hex
    filename = file_storage.filename or f"upload-{upload_id}.png"
    safe_name = filename.replace("/", "_").replace("\\", "_")
    destination = UPLOAD_DIR / f"{upload_id}_{safe_name}"
    file_storage.save(destination)
    return destination


@app.route("/", methods=["GET", "POST"])
def index():
    context: Dict[str, Any] = {
        "results": None,
        "query": "",
        "top_k": 8,
        "debug_enabled": False,
        "follow_top_k": 8,
        "mode": None,
    }

    if request.method == "POST":
        action = request.form.get("action")
        top_k = int(request.form.get("top_k", 8))
        debug_enabled = request.form.get("debug", "off") == "on"
        include_descriptions = debug_enabled

        try:
            if action == "image":
                file = request.files.get("image")
                if not file or file.filename == "":
                    raise ValueError("请先选择一张图片进行上传。")
                uploaded_path = _save_upload(file)
                service = _create_image_service()
                batch_dir = RESULTS_DIR / uuid.uuid4().hex
                results = service.recommend(
                    str(uploaded_path), top_k=top_k, destination_dir=batch_dir
                )
                payload = _build_result_payload(results, debug_enabled, include_descriptions)
                context.update(
                    {
                        "results": payload,
                        "mode": "image",
                        "top_k": top_k,
                        "debug_enabled": debug_enabled,
                        "uploaded_image": str(uploaded_path),
                    }
                )

            elif action == "text":
                query = request.form.get("query", "").strip()
                if not query:
                    raise ValueError("请输入用于检索的文本内容。")
                service = _create_text_service()
                batch_dir = RESULTS_DIR / uuid.uuid4().hex
                results = service.recommend(
                    query, top_k=top_k, destination_dir=batch_dir, include_descriptions=include_descriptions
                )
                payload = _build_result_payload(results, debug_enabled, include_descriptions)
                context.update(
                    {
                        "results": payload,
                        "mode": "text",
                        "query": query,
                        "top_k": top_k,
                        "debug_enabled": debug_enabled,
                    }
                )

            elif action == "follow":
                selected_image = request.form.get("selected_image")
                if not selected_image:
                    raise ValueError("请选择要继续检索的图片。")
                service = _create_text_service()
                batch_dir = RESULTS_DIR / uuid.uuid4().hex
                results = service.recommend_from_image(
                    selected_image,
                    top_k=top_k,
                    destination_dir=batch_dir,
                    include_descriptions=include_descriptions,
                )
                payload = _build_result_payload(results, debug_enabled, include_descriptions)
                context.update(
                    {
                        "results": payload,
                        "mode": "follow",
                        "top_k": top_k,
                        "debug_enabled": debug_enabled,
                        "follow_source": selected_image,
                    }
                )
            else:
                raise ValueError("无法识别的操作类型。")
        except Exception as exc:  # pylint: disable=broad-except
            flash(str(exc))
            context.update(
                {
                    "top_k": top_k,
                    "debug_enabled": debug_enabled,
                    "query": request.form.get("query", ""),
                }
            )

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)
