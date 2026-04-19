import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("lung_model_iqoth.keras")

classes = ['Normal', 'Benign', 'Malignant']

TEST_PATH = "dataset/test"

for img_name in os.listdir(TEST_PATH):
    img_path = os.path.join(TEST_PATH, img_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = model.predict(img)
    class_idx = np.argmax(pred)

    print(f"{img_name} → {classes[class_idx]}")