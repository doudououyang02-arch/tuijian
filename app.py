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
from Qwen3_infer import QwenTextRecommender
from text_recommand import (
    DEFAULT_MODEL_PATH as DEFAULT_TEXT_MODEL,
    DEFAULT_PATH_MATRIX,
    DEFAULT_PATH_MATRIX_ROUND1,
    DEFAULT_PATH_MATRIX_ROUND2,
    DEFAULT_PATH_MATRIX_ROUND3,
    DEFAULT_TEXT_EMBEDDINGS,
    DEFAULT_TEXT_EMBEDDINGS_ROUND1,
    DEFAULT_TEXT_EMBEDDINGS_ROUND2,
    DEFAULT_TEXT_EMBEDDINGS_ROUND3,
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
text_recommenders: Dict[str, TextRecommender] = {}

TEXT_MODELS = {
    "bge-m3": "BGE-M3",
    "qwen3": "Qwen3",
}


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


def _resolve_round_assets(search_round: int) -> Dict[str, str]:
    """Pick embeddings and path matrix according to the current round."""

    if search_round <= 1:
        return {
            "embeddings": os.getenv("TEXT_EMBEDDINGS_PATH")
            or DEFAULT_TEXT_EMBEDDINGS_ROUND1,
            "paths": os.getenv("TEXT_PATHS_MATRIX") or DEFAULT_PATH_MATRIX_ROUND1,
        }
    if search_round == 2:
        return {
            "embeddings": DEFAULT_TEXT_EMBEDDINGS_ROUND2,
            "paths": DEFAULT_PATH_MATRIX_ROUND2,
        }
    return {
        "embeddings": DEFAULT_TEXT_EMBEDDINGS_ROUND3,
        "paths": DEFAULT_PATH_MATRIX_ROUND3,
    }


def _create_text_service(
    model_choice: str, embeddings_path: str, paths_path: str
) -> TextRecommender:
    cache_key = f"{model_choice}:{embeddings_path}:{paths_path}"
    if cache_key in text_recommenders:
        return text_recommenders[cache_key]

    if model_choice == "qwen3":
        service: TextRecommender = QwenTextRecommender(
            embeddings_path=embeddings_path,
            image_paths_path=paths_path,
        )
    else:
        model_path = os.getenv("TEXT_MODEL_PATH") or DEFAULT_TEXT_MODEL
        service = TextRecommender(
            model_path=model_path,
            embeddings_path=embeddings_path,
            image_paths_path=paths_path,
        )

    text_recommenders[cache_key] = service
    return service


def _path_to_static_url(path: Path) -> str:
    """Convert a local file path under ``static`` into a browser URL."""

    static_root = STATIC_ROOT.resolve()
    display_path = Path(path)
    if display_path.exists():
        try:
            relative = display_path.resolve().relative_to(static_root)
            return url_for("static", filename=str(relative).replace(os.sep, "/"))
        except ValueError:
            return display_path.as_posix()
    return display_path.as_posix()


def _build_result_payload(
    results: List[RecommendationResult],
    debug_enabled: bool,
    include_descriptions: bool = False,
    description_service: Optional[TextRecommender] = None,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    text_service: Optional[TextRecommender] = description_service

    for item in results:
        image_url = _path_to_static_url(Path(item.display_path))

        description: Optional[str] = None
        if include_descriptions:
            if isinstance(item, TextRecommendationResult):
                description = item.description
            else:
                if text_service is None:
                    text_service = _create_text_service(
                        "bge-m3",
                        DEFAULT_TEXT_EMBEDDINGS,
                        DEFAULT_PATH_MATRIX,
                    )
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
        "mode": None,
        "current_round": None,
        "next_round": 1,
        "uploaded_image_url": None,
        "text_model": "bge-m3",
        "text_models": TEXT_MODELS,
    }

    if request.method == "POST":
        action = request.form.get("action", "search")
        top_k = int(request.form.get("top_k", 8))
        debug_enabled = request.form.get("debug", "off") == "on"
        include_descriptions = debug_enabled
        text_model = request.form.get("text_model", "bge-m3")

        # Starting a search always resets the round counter.
        search_round = 1 if action == "search" else int(request.form.get("search_round", 1))

        context.update(
            {
                "top_k": top_k,
                "debug_enabled": debug_enabled,
                "next_round": search_round,
                "text_model": text_model,
            }
        )

        try:
            if action == "follow":
                selected_image = request.form.get("selected_image")
                if not selected_image:
                    raise ValueError("请选择要继续检索的图片。")
                assets = _resolve_round_assets(search_round)
                service = _create_text_service(text_model, assets["embeddings"], assets["paths"])
                batch_dir = RESULTS_DIR / uuid.uuid4().hex
                results = service.recommend_from_image(
                    selected_image,
                    top_k=top_k,
                    destination_dir=batch_dir,
                    include_descriptions=include_descriptions,
                )
                payload = _build_result_payload(
                    results, debug_enabled, include_descriptions, description_service=service
                )
                context.update(
                    {
                        "results": payload,
                        "mode": "follow",
                        "debug_enabled": debug_enabled,
                        "follow_source": selected_image,
                        "current_round": search_round,
                        "next_round": search_round + 1,
                        "uploaded_image_url": None,
                        "text_model": text_model,
                    }
                )
                flash(f"第{search_round}轮：基于图像继续检索完成。")
            else:
                query = request.form.get("query", "").strip()
                file = request.files.get("image")
                has_text = bool(query)
                has_image = bool(file and file.filename)

                if has_image and not has_text:
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
                            "uploaded_image": str(uploaded_path),
                            "uploaded_image_url": _path_to_static_url(uploaded_path),
                            "current_round": search_round,
                            "next_round": search_round + 1,
                            "query": "",
                            "text_model": text_model,
                        }
                    )
                    flash(f"第{search_round}轮：基于上传图片完成检索。")
                elif has_text:
                    if has_image:
                        flash("当前暂不支持图片与文本联合检索，已优先使用文本进行检索。")
                    assets = _resolve_round_assets(search_round)
                    service = _create_text_service(text_model, assets["embeddings"], assets["paths"])
                    batch_dir = RESULTS_DIR / uuid.uuid4().hex
                    results = service.recommend(
                        query,
                        top_k=top_k,
                        destination_dir=batch_dir,
                        include_descriptions=include_descriptions,
                    )
                    payload = _build_result_payload(
                        results, debug_enabled, include_descriptions, description_service=service
                    )
                    context.update(
                        {
                            "results": payload,
                            "mode": "text",
                            "query": query,
                            "current_round": search_round,
                            "next_round": search_round + 1,
                            "uploaded_image_url": None,
                            "text_model": text_model,
                        }
                    )
                    flash(f"第{search_round}轮：基于文本完成检索。")
                else:
                    raise ValueError("请输入文本内容或上传图片后再发起检索。")
        except Exception as exc:  # pylint: disable=broad-except
            flash(str(exc))
            context.update(
                {
                    "query": request.form.get("query", ""),
                    "next_round": search_round,
                    "text_model": text_model,
                }
            )

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)
