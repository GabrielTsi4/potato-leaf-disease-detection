from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from pathlib import Path
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

# ----------------- ΡΥΘΜΙΣΕΙΣ -----------------
APP_IMG_SIZE = (224, 224)
CLASS_MAP = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# ----------------- FASTAPI APP -----------------
app = FastAPI(title="Potato Leaf Disease API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- FOCAL LOSS -----------------
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        w = alpha * tf.pow(1 - y_pred, gamma)
        loss = w * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

# ----------------- MODEL -----------------
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "best_model (1).keras"

print(">>> Loading model from:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

MODEL = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"focal_loss_fixed": focal_loss()}
)

print(">>> Model loaded OK")

# ----------------- GRAD-CAM SETUP -----------------
LAST_CONV_LAYER_NAME = "top_conv"

GRAD_MODEL = None  # χτίζεται στο πρώτο request

def _build_grad_model(model: tf.keras.Model):
    last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )
    return grad_model

# ----------------- PREPROCESSING -----------------
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    
    """
    img = pil_img.convert("RGB").resize(APP_IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0 # [0, 255]
    return np.expand_dims(arr, axis=0)        # (1, 224, 224, 3)

# ----------------- GRAD-CAM -----------------
def make_gradcam_overlay(pil_img: Image.Image, x_batch: np.ndarray, preds: np.ndarray) -> str:
    global GRAD_MODEL
    if GRAD_MODEL is None:
        print(">>> Building Grad-CAM model on first request...")
        GRAD_MODEL = _build_grad_model(MODEL)
        print(">>> Grad-CAM model built OK — using layer:", LAST_CONV_LAYER_NAME)

    x_tensor = tf.cast(x_batch, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        outputs = GRAD_MODEL(x_tensor)
        last_conv_output = outputs[0]
        predictions = outputs[1]
        # Αν είναι λίστα/tuple, πάρε το πρώτο στοιχείο
        if isinstance(predictions, (list, tuple)):
            predictions = tf.convert_to_tensor(predictions[0])
            if len(predictions.shape) == 1:
                predictions = tf.expand_dims(predictions, 0)
        # pred_index: scalar int
        pred_index = int(np.argmax(predictions.numpy()[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_output)  # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)

    last_conv_output = last_conv_output[0]                   # (H, W, C)
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                            # (H, W)

    # ReLU + κανονικοποίηση [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()                                # (H, W)

    # Colormap jet → RGB
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]                  # (H, W, 3)

    # Resize στο APP_IMG_SIZE
    import cv2
    jet_heatmap_resized = cv2.resize(jet_heatmap, APP_IMG_SIZE)
    jet_heatmap_uint8 = np.uint8(jet_heatmap_resized * 255)

    # Αρχική εικόνα
    base_arr = np.asarray(pil_img.convert("RGB").resize(APP_IMG_SIZE)).astype(np.uint8)

    # Blend: 60% original + 40% heatmap
    superimposed = np.clip(jet_heatmap_uint8 * 0.4 + base_arr * 0.6, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(superimposed)

    # Επιστροφή ως base64 data URL
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ----------------- ENDPOINTS -----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=415, detail="Only JPEG/PNG supported")

    try:
        data = await file.read()
        pil_img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        x = preprocess(pil_img)
        preds = MODEL.predict(x, verbose=0)[0]
        top = int(np.argmax(preds))

        gradcam_img_url = make_gradcam_overlay(pil_img, x, preds)

        return {
            "label": CLASS_MAP[top],
            "confidence": float(preds[top]),
            "probs": {CLASS_MAP[i]: float(p) for i, p in enumerate(preds)},
            "gradcam_image": gradcam_img_url,
        }
    except Exception as e:
        import traceback
        print("============ TRACEBACK ============")
        traceback.print_exc()
        print("ERROR in /predict:", repr(e))
        print("===================================")
        raise HTTPException(status_code=500, detail="Internal error during prediction")