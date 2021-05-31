from fastapi import FastAPI
from FandV_recognition.trainer import load_model, to_array, predict

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict/")
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
