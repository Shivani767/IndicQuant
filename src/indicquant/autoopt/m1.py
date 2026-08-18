"""M1 image preprocess (AutoOpt §3.1.1) and optional Tesseract OCR."""

from __future__ import annotations

from typing import Any


def preprocess_image(raw: bytes, *, width: int = 768, height: int = 1024) -> bytes:
    """Return a PNG. Requires Pillow. Used only when the user uploads an image."""
    from io import BytesIO

    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    img = Image.open(BytesIO(raw)).convert("RGB")
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    ox = (width - img.width) // 2
    oy = (height - img.height) // 2
    canvas.paste(img, (ox, oy))
    canvas = ImageOps.autocontrast(canvas)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.25)
    canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
    out = BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def ocr_png(png: bytes) -> dict[str, Any]:
    """Tesseract on an already-preprocessed PNG. Optional — not the paper's 393M model."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "M1 needs Tesseract on PATH, or paste LaTeX/text (golden-set path). "
            "We do not ship trained AutoOpt-M1 weights."
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.png"
        path.write_bytes(png)
        proc = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "eng+hin", "--psm", "6"],
            capture_output=True,
            text=True,
            check=False,
        )
    text = (proc.stdout or "").strip()
    if not text:
        raise RuntimeError("M1 produced empty OCR")
    return {"latex": text, "source": "tesseract", "chars": len(text)}


def mer_from_image(raw: bytes) -> dict[str, Any]:
    """Paper §3.1.1 preprocess, then OCR."""
    return ocr_png(preprocess_image(raw))


def mer_from_text(text: str) -> dict[str, Any]:
    """Golden-set / paste path: the image already became text (paper's M1 output)."""
    body = text.strip()
    if not body:
        raise ValueError("empty formulation")
    return {"latex": body, "source": "text", "chars": len(body)}
