import numpy as np
from keras.src.saving import load_model

model = load_model('../models/clear_model_epoch_3_15.keras')

# Словарь эмоций
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}


def predict_emotion(image_gray):
    prediction = model.predict(image_gray)
    print(prediction)
    max_index = int(np.argmax(prediction))
    predicted_emotion = emotion_dict[max_index]
    return predicted_emotion
