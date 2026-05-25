import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model


#loading model
model = load_model("lung_model_iqoth.keras")

gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

val_data = gen.flow_from_directory(
    "dataset/train",
    target_size=(260, 260),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False,          
)


print("Class Indices:")
print(val_data.class_indices)

labels = [k for k, v in sorted(val_data.class_indices.items(), key=lambda x: x[1])]


#predict
predictions = model.predict(val_data, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))


# confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved -> confusion_matrix.png")
