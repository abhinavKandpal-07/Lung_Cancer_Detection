import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

#Loading model
model = load_model("lung_model_iqoth.keras")

#Loading data
gen = ImageDataGenerator(rescale=1./255)

data = gen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

#Predict
predictions = model.predict(data)

y_pred = np.argmax(predictions, axis=1)
y_true = data.classes

#Labels
labels = list(data.class_indices.keys())


print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))