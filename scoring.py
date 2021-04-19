import numpy as np
import json
import joblib
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model.pkl')
    model = joblib.load(model_path)

def run(features):
    try:
        features = np.array(json.loads(features))
        predictions = model.predict(features)
        return predictions.tolist()
    except Exception as exp:
        pass