import os
import json
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load Model
model = load_model("lung_model_iqoth.keras")

# Load Class Labels
CLASS_INDEX_FILE = "class_indices.json"

if os.path.exists(CLASS_INDEX_FILE):
    with open(CLASS_INDEX_FILE) as f:
        class_indices = json.load(f)
    classes = {v: k for k, v in class_indices.items()}
else:
    print(
        f"[WARNING] {CLASS_INDEX_FILE} not found. "
        "Using hardcoded fallback. Run save_class_indices() first."
    )
    classes = {0: "Benign", 1: "Malignant", 2: "Normal"}

TEST_PATH = "dataset/test"

# Predict Images
results = []

for img_name in sorted(os.listdir(TEST_PATH)):
    img_path = os.path.join(TEST_PATH, img_name)

    # skip if these format are there
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Could not read: {img_name}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (260, 260))

    img_array = np.expand_dims(img.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array, verbose=0)[0]
    class_idx  = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    label      = classes.get(class_idx, str(class_idx))

    print(f"{img_name:<40} → {label:<12} ({confidence:.2f}%)")
    results.append({"file": img_name, "label": label, "confidence": confidence})

print(f"\nProcessed {len(results)} image(s).")


def save_class_indices(train_data, path: str = "class_indices.json") -> None:
    """Save train_data.class_indices to JSON so predict.py can load them."""
    with open(path, "w") as f:
        json.dump(train_data.class_indices, f, indent=2)
    print(f"Class indices saved → {path}")
