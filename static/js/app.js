(function () {
  function ready(fn) {
    if (document.readyState !== "loading") {
      fn();
    } else {
      document.addEventListener("DOMContentLoaded", fn);
    }
  }

  ready(function () {
    const modal = document.getElementById("image-modal");
    if (!modal) {
      return;
    }

    const debugEnabled = modal.dataset.debug === "true";
    const modalImage = document.getElementById("modal-image");
    const modalFilename = document.getElementById("modal-filename");
    const modalDescription = document.getElementById("modal-description");

    function closeModal() {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
      if (modalImage) {
        modalImage.removeAttribute("src");
      }
      if (modalDescription) {
        modalDescription.textContent = "";
        modalDescription.style.display = "none";
      }
    }

    document.querySelectorAll('[data-role="close-modal"]').forEach(function (btn) {
      btn.addEventListener("click", closeModal);
    });

    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal();
      }
    });

    document.querySelectorAll('[data-role="preview"]').forEach(function (button) {
      button.addEventListener("click", function () {
        const imageUrl = button.dataset.url;
        const filename = button.dataset.filename || "";
        let description = button.dataset.description || "";
        if (description) {
          try {
            description = JSON.parse(description);
          } catch (error) {
            // Ignore JSON parse errors and fall back to raw text.
          }
        }

        if (modalImage && imageUrl) {
          modalImage.src = imageUrl;
        }
        if (modalFilename) {
          modalFilename.textContent = filename;
        }
        if (modalDescription) {
          if (debugEnabled && description) {
            modalDescription.textContent = description;
            modalDescription.style.display = "block";
          } else {
            modalDescription.textContent = "";
            modalDescription.style.display = "none";
          }
        }

        modal.classList.add("open");
        modal.setAttribute("aria-hidden", "false");
      });
    });

    const followForm = document.getElementById("follow-form");
    const followImageInput = document.getElementById("follow-image-input");

    document.querySelectorAll('[data-role="next-round"]').forEach(function (button) {
      button.addEventListener("click", function () {
        if (!followForm || !followImageInput) {
          return;
        }
        followImageInput.value = button.dataset.source || "";
        followForm.submit();
      });
    });
  });
})();
