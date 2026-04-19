from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

#earlier model overfittted
def build_model():
    model = Sequential([

        Input(shape=(224,224,3)),

        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(0.5),   

        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model