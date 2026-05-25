from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def build_model(num_classes: int = 3) -> Model:

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(260, 260, 3),
    )

    # Freeze backbone initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    return model


def unfreeze_top_layers(model: Model, num_layers: int = 15) -> Model:

    # Unfreeze only top layers
    for layer in model.layers[-num_layers:]:

        # Keep BatchNorm frozen
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(
            learning_rate=1e-5
        ),

        loss=CategoricalCrossentropy(label_smoothing=0.1),

        metrics=["accuracy"],
    )

    print(f"\nUnfroze last {num_layers} layers.")

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"Trainable layers: {trainable}/{len(model.layers)}")

    return model
