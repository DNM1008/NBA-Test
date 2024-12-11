# model.py
import pickle  # Example for loading your model

import dill
from annoy import AnnoyIndex

# Load your trained model
# model = pickle.load(open("collaborative_search_model_with_parallel.pkl", "rb"))

annoy_index = AnnoyIndex(f, "angular")
annoy_index.load("annoy_index.ann")

# def predict(data):
#    """
#    Predict using the pre-trained model.
#    """
#    prediction = model.predict([data])  # Adjust this to your model's input requirements
#    return prediction[0]

# Load the other components
with open("collaborative_search_model_with_parallel.pkl", "rb") as f:
    saved_objects = dill.load(f)

# Access the loaded objects
transformer = saved_objects["transformer"]
y_train = saved_objects["y_train"]
predict = saved_objects["predict_nba_parallel"]
