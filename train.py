import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,)
from model import build_model, unfreeze_top_layers

TRAIN_PATH = "dataset/train"

IMG_SIZE = (260, 260)

BATCH_SIZE = 16
NUM_CLASSES = 3


# data generators
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,

    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.08,

    horizontal_flip=True,
    fill_mode="nearest",
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

train_data = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_data = val_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)

print("\nSaved class_indices.json")


print("\nClass Indices:")
print(train_data.class_indices)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes,
)

class_weights = dict(enumerate(class_weights))

class_weights[0] *= 1.8   # benign
class_weights[2] *= 1.3   # normal

print("\nAdjusted Class Weights:")
print(class_weights)

# model build
model = build_model(
    num_classes=NUM_CLASSES
)

model.summary()

# checkpoint
checkpoint = ModelCheckpoint(
    "best_lung_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

# 1st phase
print("\n" + "=" * 50)
print("PHASE 1: Training Head")
print("=" * 50)

early_stop_1 = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr_1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1,
)

history1 = model.fit(
    train_data,
    validation_data=val_data,

    epochs=15,

    callbacks=[
        early_stop_1,
        reduce_lr_1,
        checkpoint,
    ],

    class_weight=class_weights,
)


# 2nd phase
print("\n" + "=" * 50)
print("PHASE 2: Fine-Tuning")
print("=" * 50)

model = unfreeze_top_layers(
    model,
    num_layers=15,
)

early_stop_2 = EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr_2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1,
)

history2 = model.fit(
    train_data,
    validation_data=val_data,

    epochs=15,

    callbacks=[
        early_stop_2,
        reduce_lr_2,
        checkpoint,
    ],

    class_weight=class_weights,
)


model.save(
    "lung_model_iqoth.keras"
)

print("\nModel saved -> lung_model_iqoth.keras")


# plot history
phase1_epochs = len(history1.history["accuracy"])
phase2_epochs = len(history2.history["accuracy"])

total_epochs = phase1_epochs + phase2_epochs

x1 = range(1, phase1_epochs + 1)
x2 = range(phase1_epochs + 1, total_epochs + 1)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)


# ACCURACY
ax = axes[0]

ax.plot(x1,history1.history["accuracy"],label="Train Phase 1")
ax.plot(x1,history1.history["val_accuracy"],linestyle="--",label="Val Phase 1")

ax.plot(x2,history2.history["accuracy"],label="Train Phase 2")
ax.plot(x2,history2.history["val_accuracy"],linestyle="--",label="Val Phase 2")

ax.axvline(x=phase1_epochs,linestyle=":",label="Fine-Tuning Start")

ax.set_title("Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()

# LOSS
ax = axes[1]

ax.plot(x1,history1.history["loss"],label="Train Phase 1")
ax.plot(x1,history1.history["val_loss"],linestyle="--",label="Val Phase 1")

ax.plot(x2,history2.history["loss"], label="Train Phase 2")
ax.plot(x2,history2.history["val_loss"],linestyle="--",label="Val Phase 2")

ax.axvline(x=phase1_epochs, linestyle=":", label="Fine-Tuning Start")

ax.set_title("Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

plt.tight_layout()
plt.savefig("training_history.png",dpi=150,)
plt.show()

print("Training plot saved -> training_history.png")







