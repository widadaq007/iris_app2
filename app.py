from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read features from HTML form
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        classes = ["Setosa", "Versicolor", "Virginica"]

        return render_template("index.html", prediction_text=f"🌸 Predicted Iris: {classes[prediction]}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)