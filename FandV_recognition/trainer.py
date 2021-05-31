from tensorflow import keras
import os
import cv2
import numpy as np

LABELS = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon']


def load_model():
    model = keras.models.load_model('model/model.keras')
    return model


def to_array(image):
    image_array = cv2.imread(image, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_array, (100, 100))
    return image_resized


def predict(image):
    print(">>> Loading the model...")
    model = load_model()
    print(">>> Loading successful !")
    print(">>> image to array... ")
    image = to_array(image)
    print(">>> Transformation successful !")
    print(">>> Predicting...")
    image_reshaped = image.reshape(-1, 100, 100, 3)
    index_pred = np.argmax(model.predict(image_reshaped), axis=-1)[0]
    prediction = LABELS[index_pred]
    if prediction in ['apple', 'eggplant', 'onion', 'orange']:
        return print(f" This image is an {prediction}")
    return print(f" This image is a {prediction}")


if __name__ == "__main__":
    image_test = f"raw_data/test/apple/Image_3.jpg"
    predict(image_test)
