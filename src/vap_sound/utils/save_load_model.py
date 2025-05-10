import pickle

# Function to save model as pickle file
def save_model_pickle(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully as pickle file!")

# Function to load model from pickle file
def load_model_pickle(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    model.eval()
    print("Model loaded successfully from pickle file!")
    return model