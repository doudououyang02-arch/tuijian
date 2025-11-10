(function () {
  function ready(fn) {
    if (document.readyState !== "loading") {
      fn();
    } else {
      document.addEventListener("DOMContentLoaded", fn);
    }
  }

  function parseDescription(raw) {
    if (!raw || raw === "null") {
      return "";
    }
    let parsed = raw;
    try {
      parsed = JSON.parse(raw);
      if (typeof parsed === "string") {
        return parsed;
      }
      return JSON.stringify(parsed, null, 2);
    } catch (error) {
      return raw;
    }
  }

  ready(function () {
    const dropzone = document.getElementById("search-dropzone");
    const imageInput = document.getElementById("image-input");
    const uploadTrigger = document.getElementById("upload-trigger");
    const queryInput = document.getElementById("query-input");
    const topKInput = document.querySelector('#search-form input[name="top_k"]');
    const thumbnailWrapper = dropzone ? dropzone.querySelector('[data-role="thumbnail"]') : null;
    const thumbnailImage = thumbnailWrapper ? thumbnailWrapper.querySelector("img") : null;
    const thumbnailRemove = thumbnailWrapper
      ? thumbnailWrapper.querySelector('[data-role="remove-image"]')
      : null;

    function setHasImage(enabled) {
      if (!dropzone) {
        return;
      }
      dropzone.classList.toggle("has-image", Boolean(enabled));
      if (thumbnailRemove) {
        thumbnailRemove.style.display = enabled ? "flex" : "none";
      }
    }

    function clearThumbnail() {
      if (!thumbnailImage) {
        return;
      }
      thumbnailImage.removeAttribute("src");
      thumbnailImage.hidden = true;
      setHasImage(false);
      if (imageInput) {
        imageInput.value = "";
      }
    }

    function updateThumbnailFromFile(file) {
      if (!file || !thumbnailImage) {
        return;
      }
      const reader = new FileReader();
      reader.onload = function (event) {
        if (!event.target) {
          return;
        }
        const result = event.target.result;
        if (typeof result === "string") {
          thumbnailImage.src = result;
          thumbnailImage.hidden = false;
          setHasImage(true);
        }
      };
      reader.readAsDataURL(file);
    }

    function updateThumbnailFromUrl(url) {
      if (!thumbnailImage || !url) {
        return;
      }
      thumbnailImage.src = url;
      thumbnailImage.hidden = false;
      setHasImage(true);
    }

    if (dropzone) {
      if (queryInput) {
        const syncTextState = function () {
          dropzone.classList.toggle("has-text", queryInput.value.trim().length > 0);
        };
        queryInput.addEventListener("input", syncTextState);
        syncTextState();
      }

      dropzone.addEventListener("dragover", function (event) {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
      dropzone.addEventListener("dragleave", function (event) {
        if (event.target === dropzone) {
          dropzone.classList.remove("dragover");
        }
      });
      dropzone.addEventListener("drop", function (event) {
        event.preventDefault();
        dropzone.classList.remove("dragover");
        const files = event.dataTransfer ? event.dataTransfer.files : null;
        if (files && files.length > 0 && imageInput) {
          const fileList = new DataTransfer();
          fileList.items.add(files[0]);
          imageInput.files = fileList.files;
          updateThumbnailFromFile(files[0]);
        }
      });
      dropzone.addEventListener("click", function (event) {
        if (event.target === dropzone && queryInput) {
          queryInput.focus();
        }
      });

      const serverPreview = dropzone.dataset.uploadPreview;
      if (serverPreview) {
        updateThumbnailFromUrl(serverPreview);
      }
    }

    if (imageInput) {
      imageInput.addEventListener("change", function () {
        const file = imageInput.files && imageInput.files[0];
        if (file) {
          updateThumbnailFromFile(file);
        } else {
          clearThumbnail();
        }
      });
    }

    if (uploadTrigger && imageInput) {
      uploadTrigger.addEventListener("click", function () {
        imageInput.click();
      });
    }

    if (thumbnailRemove) {
      thumbnailRemove.addEventListener("click", function (event) {
        event.stopPropagation();
        clearThumbnail();
      });
    }

    const modal = document.getElementById("image-modal");
    if (!modal) {
      return;
    }

    const debugEnabled = modal.dataset.debug === "true";
    const modalImage = document.getElementById("modal-image");
    const modalJson = document.getElementById("modal-json");
    const modalFilename = document.getElementById("modal-filename");
    const modalScore = document.getElementById("modal-score");
    const modalSource = document.getElementById("modal-source");
    const modalZoomContainer = document.getElementById("modal-zoom-container");

    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let originTranslateX = 0;
    let originTranslateY = 0;
    let naturalWidth = 0;
    let naturalHeight = 0;

    function resetZoom() {
      scale = 1;
      translateX = 0;
      translateY = 0;
      applyTransform();
    }

    function applyTransform() {
      if (!modalImage) {
        return;
      }
      clampTranslations();
      modalImage.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    }

    function getBaseSize() {
      if (!modalZoomContainer) {
        return { width: 0, height: 0 };
      }
      const containerWidth = modalZoomContainer.clientWidth;
      const containerHeight = modalZoomContainer.clientHeight;
      if (!naturalWidth || !naturalHeight) {
        return { width: containerWidth, height: containerHeight };
      }
      const imageAspect = naturalWidth / naturalHeight;
      const containerAspect = containerWidth / containerHeight || 1;
      if (containerAspect > imageAspect) {
        const height = containerHeight;
        return { height, width: height * imageAspect };
      }
      const width = containerWidth;
      return { width, height: width / imageAspect };
    }

    function clampTranslations() {
      if (!modalZoomContainer) {
        return;
      }
      const containerWidth = modalZoomContainer.clientWidth;
      const containerHeight = modalZoomContainer.clientHeight;
      const baseSize = getBaseSize();
      const scaledWidth = baseSize.width * scale;
      const scaledHeight = baseSize.height * scale;
      const maxX = Math.max((scaledWidth - containerWidth) / 2, 0);
      const maxY = Math.max((scaledHeight - containerHeight) / 2, 0);
      translateX = Math.min(Math.max(translateX, -maxX), maxX);
      translateY = Math.min(Math.max(translateY, -maxY), maxY);
    }

    function closeModal() {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
      if (modalImage) {
        modalImage.removeAttribute("src");
        modalImage.style.transform = "";
      }
      if (modalJson) {
        modalJson.textContent = "";
        modalJson.style.display = "none";
      }
      if (modalFilename) {
        modalFilename.textContent = "";
      }
      if (modalScore) {
        modalScore.textContent = "";
      }
      if (modalSource) {
        modalSource.textContent = "";
      }
      resetZoom();
    }

    function openModal(data) {
      if (!modalImage) {
        return;
      }
      resetZoom();
      if (data.url) {
        modalImage.src = data.url;
      }
      modalImage.style.transformOrigin = "center";
      if (modalFilename) {
        modalFilename.textContent = data.filename || "";
      }
      if (modalScore) {
        modalScore.textContent = data.score ? `相似度 ${data.score}` : "";
      }
      if (modalSource) {
        modalSource.textContent = data.source || "";
      }
      if (debugEnabled && modalJson) {
        const description = parseDescription(data.description || "");
        if (description) {
          modalJson.textContent = description;
          modalJson.style.display = "block";
        } else {
          modalJson.textContent = "";
          modalJson.style.display = "none";
        }
      } else if (modalJson) {
        modalJson.textContent = "";
        modalJson.style.display = "none";
      }
      modal.classList.add("open");
      modal.setAttribute("aria-hidden", "false");
    }

    if (modalImage) {
      modalImage.addEventListener("load", function () {
        naturalWidth = modalImage.naturalWidth || modalImage.width || 0;
        naturalHeight = modalImage.naturalHeight || modalImage.height || 0;
        resetZoom();
      });
    }

    document.querySelectorAll('[data-role="close-modal"]').forEach(function (btn) {
      btn.addEventListener("click", closeModal);
    });

    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal();
      }
    });

    if (modalZoomContainer) {
      modalZoomContainer.addEventListener("wheel", function (event) {
        if (!modalImage) {
          return;
        }
        event.preventDefault();
        const delta = event.deltaY;
        const zoomFactor = delta < 0 ? 1.1 : 0.9;
        const newScale = Math.min(Math.max(scale * zoomFactor, 1), 6);
        if (Math.abs(newScale - scale) < 0.001) {
          return;
        }
        const rect = modalZoomContainer.getBoundingClientRect();
        const pointerX = event.clientX - rect.left - rect.width / 2;
        const pointerY = event.clientY - rect.top - rect.height / 2;
        translateX -= pointerX * (newScale / scale - 1);
        translateY -= pointerY * (newScale / scale - 1);
        scale = newScale;
        applyTransform();
      });

      modalZoomContainer.addEventListener("pointerdown", function (event) {
        if (scale <= 1) {
          return;
        }
        isDragging = true;
        dragStartX = event.clientX;
        dragStartY = event.clientY;
        originTranslateX = translateX;
        originTranslateY = translateY;
        modalZoomContainer.setPointerCapture(event.pointerId);
      });

      modalZoomContainer.addEventListener("pointermove", function (event) {
        if (!isDragging) {
          return;
        }
        translateX = originTranslateX + (event.clientX - dragStartX);
        translateY = originTranslateY + (event.clientY - dragStartY);
        applyTransform();
      });

      function endDrag(event) {
        if (!isDragging) {
          return;
        }
        isDragging = false;
        modalZoomContainer.releasePointerCapture(event.pointerId);
      }

      modalZoomContainer.addEventListener("pointerup", endDrag);
      modalZoomContainer.addEventListener("pointerleave", endDrag);
      modalZoomContainer.addEventListener("pointercancel", endDrag);
    }

    function collectDataFromElement(element) {
      if (!element) {
        return null;
      }
      return {
        url: element.dataset.url || element.getAttribute("src") || "",
        description: element.dataset.description || "",
        filename: element.dataset.filename || "",
        score: element.dataset.score || "",
        source: element.dataset.source || "",
      };
    }

    document.querySelectorAll('[data-role="open-modal"]').forEach(function (button) {
      button.addEventListener("click", function (event) {
        event.stopPropagation();
        const data = collectDataFromElement(button);
        if (data) {
          openModal(data);
        }
      });
    });

    document.querySelectorAll(".masonry-item img").forEach(function (img) {
      img.addEventListener("click", function () {
        const data = collectDataFromElement(img);
        if (data) {
          openModal(data);
        }
      });
    });

    const followForm = document.getElementById("follow-form");
    const followImageInput = document.getElementById("follow-image-input");
    const followTopkInput = document.getElementById("follow-topk-input");

    if (topKInput && followTopkInput) {
      topKInput.addEventListener("input", function () {
        followTopkInput.value = topKInput.value;
      });
      followTopkInput.addEventListener("input", function () {
        topKInput.value = followTopkInput.value;
      });
    }

    document.querySelectorAll('[data-role="continue-search"]').forEach(function (button) {
      button.addEventListener("click", function (event) {
        event.stopPropagation();
        if (!followForm || !followImageInput) {
          return;
        }
        followImageInput.value = button.dataset.source || "";
        if (topKInput && followTopkInput) {
          followTopkInput.value = topKInput.value;
        }
        followForm.submit();
      });
    });
  });
})();
