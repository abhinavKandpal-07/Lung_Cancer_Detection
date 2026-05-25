import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


# load model
model = load_model("lung_model_iqoth.keras")

CLASS_INDEX_FILE = "class_indices.json"

if os.path.exists(CLASS_INDEX_FILE):
    with open(CLASS_INDEX_FILE) as f:
        class_indices = json.load(f)
    classes = {v: k for k, v in class_indices.items()}
else:
    classes = {0: "Benign", 1: "Malignant", 2: "Normal"}

CLASS_COLORS = {
    "Benign":     "#28a745",   #
    "Malignant":  "#dc3545",   
    "Normal":     "#007acc",   
}


# Predict function
def predict_image():
    global file_path

    if not file_path:
        messagebox.showerror("Error", "Please upload a CT scan image first.")
        return

    #Read & preproces
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        messagebox.showerror("Error", "Could not read the image file.")
        return

    #convert BGR -> RGB (cv2 loads as BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (260, 260))

    img_array = np.expand_dims(img_rgb.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    #Prediction
    pred        = model.predict(img_array, verbose=0)[0]
    class_idx   = int(np.argmax(pred))
    confidence  = float(np.max(pred)) * 100
    label       = classes.get(class_idx, str(class_idx))

    # Show probabilities
    prob_text = "  |  ".join(
        f"{classes.get(i, i)}: {pred[i]*100:.1f}%"
        for i in range(len(pred))
    )

    color = CLASS_COLORS.get(label, "#333333")

    result_label.config(
        text=f"Prediction: {label}  ({confidence:.1f}%)",
        fg=color,
        font=("Arial", 16, "bold"),
    )
    prob_label.config(
        text=prob_text,
        fg="#555555",
        font=("Arial", 10),
    )


# Upload image function
def upload_image():
    global file_path

    path = filedialog.askopenfilename(
        title="Select CT Scan Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
    )

    if not path:
        return

    file_path = path

    # Reset result labels when a new image is loaded
    result_label.config(text="")
    prob_label.config(text="")

    img = Image.open(file_path).resize((260, 260))
    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo  

# GUI Window
root = tk.Tk()
root.title("Lung Cancer Detection System")
root.geometry("620x780")
root.configure(bg="#f0f0f0")
root.resizable(False, False)

file_path = ""

# Heading
tk.Label(
    root,
    text="Lung Cancer Detection using CNN",
    font=("Arial", 20, "bold"),
    bg="#f0f0f0",
    fg="#003366",
).pack(pady=20)


tk.Label(root, text="Patient Name", font=("Arial", 12), bg="#f0f0f0").pack()
name_entry = tk.Entry(root, width=30, font=("Arial", 12))
name_entry.pack(pady=5)

tk.Label(root, text="Patient Age", font=("Arial", 12), bg="#f0f0f0").pack()
age_entry = tk.Entry(root, width=30, font=("Arial", 12))
age_entry.pack(pady=5)

# Upload Button
tk.Button(
    root,
    text="Upload CT Scan",
    command=upload_image,
    bg="#007acc",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=10,
    pady=6,
    relief="flat",
    cursor="hand2",
).pack(pady=18)


#show image
image_label = tk.Label(root, bg="#dddddd", width=260, height=260)
image_label.pack()


#button predict
tk.Button(
    root,
    text="🔍  Predict",
    command=predict_image,
    bg="#28a745",
    fg="white",
    font=("Arial", 14, "bold"),
    padx=20,
    pady=8,
    relief="flat",
    cursor="hand2",
).pack(pady=18)


# Result labels
result_label = tk.Label(root, text="", bg="#f0f0f0")
result_label.pack(pady=6)

# probability shown of each class
prob_label = tk.Label(root, text="", bg="#f0f0f0")
prob_label.pack()

root.mainloop()