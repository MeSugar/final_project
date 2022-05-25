import pandas as pd
from fastapi import FastAPI
from helpers import generate_features
from helpers import UserRequestIn
from joblib import load


app = FastAPI()

classes = {
    1 : "Spruce/Fir",
    2 : "Lodgepole Pine",
    3 : "Ponderosa Pine",
    4 : "Cottonwood/Willow",
    5 : "Aspen",
    6 : "Douglas-fir",
    7 : "Krummholz",
}

@app.get("/")
def root():
    return {'hello'}

@app.post('/predict')
def predict(data : UserRequestIn):
    data = data.dict()
    new_data = pd.DataFrame(generate_features(data), index=['Id'])
    new_data.set_index('Id', drop=True, inplace=True)
    clf = load("model.joblib")
    predicted_value = clf.predict(new_data)
    return "Cover type is: {}".format(classes[int(predicted_value)])