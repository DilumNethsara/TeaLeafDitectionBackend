"""
Tea Disease Classification API - FastAPI + MobileNetV3

Endpoints:
  GET  /health   - backend readiness check for the mobile app
  GET  /         - simple browser UI
  POST /predict  - classify a tea leaf image
  POST /explain  - classify + return Grad-CAM heatmap overlay
"""

import base64
import io
import os

import matplotlib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.applications.mobilenet_v3 import preprocess_input
from keras.src.layers.core.dense import Dense as _KerasDense
from keras.utils import img_to_array, load_img
from tensorflow.keras.models import load_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The model was saved with a newer Keras version that serializes
# quantization_config on Dense layers. Older runtimes reject that key.
_original_dense_init = _KerasDense.__init__


def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)


_KerasDense.__init__ = _patched_dense_init

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "best_mobilenetv3.keras")

os.makedirs(STATIC_DIR, exist_ok=True)

CLASS_LABELS = ["Black Blight", "Grey Blight", "Spider Mites", "Healthy"]
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)
_base_model = model.get_layer("MobileNetV3Large")
_last_conv_name = None

for _layer in reversed(_base_model.layers):
    if isinstance(_layer, tf.keras.layers.Conv2D):
        _last_conv_name = _layer.name
        break

print(f"[startup] Last Conv2D layer: {_last_conv_name}")
print(f"[startup] Model input : {model.input_shape}")
print(f"[startup] Model output: {model.output_shape}")

app = FastAPI(title="Tea Disease Classification", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def load_and_preprocess(image_bytes: bytes):
    pil_img = load_img(io.BytesIO(image_bytes), target_size=IMG_SIZE)
    arr = img_to_array(pil_img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr, pil_img


def _predict(image_bytes):
    x, raw = load_and_preprocess(image_bytes)
    logits = model.predict(x, verbose=0)
    probs = softmax(logits)[0]
    pred_idx = int(np.argmax(probs))
    return x, raw, probs, pred_idx


def _gradcam(img_tensor, class_idx):
    conv_layer = _base_model.get_layer(_last_conv_name)
    inner = tf.keras.Model(
        inputs=_base_model.input,
        outputs=[conv_layer.output, _base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, base_out = inner(img_tensor)
        h = model.get_layer("global_average_pooling2d")(base_out)
        h = model.get_layer("dense")(h)
        h = model.get_layer("dropout")(h)
        preds = model.get_layer("dense_1")(h)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(
        conv_out[0] * pooled[0][tf.newaxis, tf.newaxis, :], axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap - tf.reduce_min(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _build_probs_dict(probs):
    return {cls: float(prob) for cls, prob in zip(CLASS_LABELS, probs)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": CLASS_LABELS,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    _, _, probs, pred_idx = _predict(contents)
    return {
        "predicted_class": CLASS_LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": _build_probs_dict(probs),
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    contents = await file.read()
    x, raw, probs, pred_idx = _predict(contents)
    heatmap = _gradcam(x, pred_idx)

    h_resized = tf.image.resize(
        np.expand_dims(heatmap, axis=-1), (raw.size[1], raw.size[0])
    ).numpy().squeeze()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(raw)
    ax.imshow(h_resized, cmap="jet", alpha=0.45)
    ax.set_title(
        f"Grad-CAM: {CLASS_LABELS[pred_idx]} ({probs[pred_idx]:.1%})",
        fontsize=11,
    )
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "predicted_class": CLASS_LABELS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": _build_probs_dict(probs),
        "gradcam_overlay": f"data:image/png;base64,{overlay_b64}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
