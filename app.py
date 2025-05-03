import json
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 

# Load the trained model
with open("earthquake_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

# Define Risk Levels & Preventions
RISK_MEASURES = [
    {
        "level": "Very Low : 0-2.9",
        "range": [0, 2.9],
        "description": "Barely noticeable.",
        "prevention": [
            "Barely noticeable",
            "Protect yourself No need for immediate action, but remain alert for any unusual signs.",
            "Ensure that small objects or furniture are secured to avoid minor shifting.",
            "Regularly check local building codes and safety measures in case of future larger tremors."
        ]
    },
    {
        "level": "Low : 3.0-4.9",
        "range": [3.0, 4.9],
        "description": "Minor tremors",
        "prevention": [
            "Minor tremors",
            "Stay indoors, ensure furniture is stable. Light shaking may occur. No need for evacuation unless a larger aftershock follows.",
            "Check that heavy furniture is anchored and that exit routes are not obstructed.",
            "Educate family members on basic earthquake safety (e.g., 'Drop, Cover, and Hold On')."
        ]
    },
    {
        "level": "Moderate : 5.0-6.9",
        "range": [5.0, 6.9],
        "description": "Felt indoors, minor damage.",
        "prevention": [
            "Felt indoors minor damage",
            "Stay away from windows & heavy objects. Take cover immediately under furniture or against an interior wall. Stay away from windows and glass doors.",
            "If indoors, remain inside and avoid running outside as debris may fall.",
            "Ensure you have an emergency kit ready, including food, water, first aid supplies, and flashlights."
        ]
    },
    {
        "level": "High : 7.0-7.9",
        "range": [7.0, 7.9],
        "description": "Significant damage.",
        "prevention": [
            "Significant damage",
            "Have an emergency kit, secure heavy objects. Evacuate unsafe buildings if possible. Drop to the ground, take cover, and hold on during shaking.",
            "Assess structural damage to the building, including gas leaks or electrical hazards.",
            "If you live in a high-risk area, prepare to move to higher ground immediately after the shaking subsides in case of a tsunami."
        ]
    },
    {
        "level": "Severe : 8.0-10.0",
        "range": [8.0, 10.0],
        "description": "Major destruction.",
        "prevention": [
            "Major destruction","Evacuate to an open space immediately. Evacuate immediately if in a building that may collapse or show visible damage.",
            "Be ready for aftershocks and stay away from potentially hazardous areas, such as bridges, damaged roads, and buildings.",
            "Implement emergency plans to access food, water, and medical supplies, and establish communication with emergency services."
        ]
    }
]

# Load history from JSON file
def load_history():
    try:
        with open("data.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Save history to JSON file
def save_history(data):
    with open("data.json", "w") as file:
        json.dump(data, file, indent=4)

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/earthquake")
def earthquake_page():
    return render_template("index.html")

@app.route("/weather")
def weather_page():
    return render_template("weather.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    magnitude = float(data["Magnitude"])
    depth = float(data["Depth"])
    latitude = float(data["Latitude"])
    longitude = float(data["Longitude"])

    # Convert input into a DataFrame with correct feature names
    feature_names = ["Magnitude", "Depth", "Latitude", "Longitude"]
    input_data = pd.DataFrame([[magnitude, depth, latitude, longitude]], columns=feature_names)

    # Predict risk level
    prediction = model.predict(input_data)[0]

    # Get corresponding risk measures
    for risk in RISK_MEASURES:
        if risk["range"][0] <= magnitude <= risk["range"][1]:
            risk_level = risk["level"]
            prevention = risk["prevention"]
            break

    response = {
        "Magnitude": magnitude,
        "Depth": depth,
        "Latitude": latitude,
        "Longitude": longitude,
        "Risk_Level": risk_level,
        "Prevention": prevention
    }

    # Store in JSON file
    history = load_history()
    history.append(response)
    save_history(history)

    return jsonify(response)

@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())

if __name__ == "__main__":
    app.run(debug=True)
