
# ===============================
# 🚌 Daewoo Bus Delay Predictor
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import datetime as dt

# 1️⃣ Load your synthetic dataset
df = pd.read_csv("synthetic_daewoo_delay_data.csv")

# 2️⃣ Encode categorical variables
encoders = {}
for col in ["Origin", "Destination", "Day", "Weather"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Convert Departure time (HH:MM) → minutes since midnight
df["Departure"] = df["Departure"].apply(lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1]))

# 3️⃣ Split into features (X) and target (y)
X = df[["Origin", "Destination", "Scheduled_Duration(min)", "Departure", "Day", "Weather"]]
y = df["Delay(min)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluate basic performance
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# =========================================
# 🧠 Real-time User Prediction
# =========================================

# Take user input
print("\n🚌 === Delay Prediction Input Form ===")
origin_input = input("Enter Origin City (e.g., Lahore): ").title()
destination_input = input("Enter Destination City (e.g., Islamabad): ").title()
date_input = input("Enter Travel Date (YYYY-MM-DD): ")
time_input = input("Enter Departure Time (HH:MM, 24h format): ")

import requests

# ------------- 🌦️ AUTO WEATHER DETECTION -------------
def get_weather_condition(city, date):
    """
    Fetch forecast from Open-Meteo API (free, no key).
    Returns simplified condition: 'Clear', 'Rainy', or 'Foggy'.
    """
    # Map major cities to lat/lon (expand as needed)
    city_coords = {
        "Lahore": (31.5204, 74.3587),
        "Islamabad": (33.6844, 73.0479),
        "Karachi": (24.8607, 67.0011),
        "Multan": (30.1575, 71.5249),
        "Faisalabad": (31.418, 73.079),
        "Peshawar": (34.0151, 71.5249),
        "Quetta": (30.1798, 66.975),
        "Rawalpindi": (33.6, 73.033),
        "Sialkot": (32.5, 74.5),
        "Jhang": (31.268, 72.318)
    }

    if city not in city_coords:
        return "Clear"  # fallback if city not listed

    lat, lon = city_coords[city]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weathercode&start_date={date}&end_date={date}&timezone=auto"
    response = requests.get(url)
    data = response.json()

    try:
        code = data["daily"]["weathercode"][0]
    except (KeyError, IndexError):
        return "Clear"

    # Map weather codes to readable labels
    if code in [0, 1]: return "Clear"
    elif code in [2, 3, 45, 48]: return "Foggy"
    elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "Rainy"
    else: return "Clear"

# Fetch automatically
weather_input = get_weather_condition(origin_input, date_input)
print(f"🌦️ Weather detected for {origin_input} on {date_input}: {weather_input}")


# Extract weekday from date
day_name = dt.datetime.strptime(date_input, "%Y-%m-%d").strftime("%A")

# Prepare features
try:
    origin_val = encoders["Origin"].transform([origin_input])[0]
except ValueError:
    origin_val = 0  # unseen city fallback

try:
    dest_val = encoders["Destination"].transform([destination_input])[0]
except ValueError:
    dest_val = 0

try:
    day_val = encoders["Day"].transform([day_name])[0]
except ValueError:
    day_val = 0

try:
    weather_val = encoders["Weather"].transform([weather_input])[0]
except ValueError:
    weather_val = 0

dep_minutes = int(time_input.split(":")[0]) * 60 + int(time_input.split(":")[1])

# Pick average scheduled duration (you can refine this later)
avg_duration = df["Scheduled_Duration(min)"].mean()

# Create feature row
input_features = [[origin_val, dest_val, avg_duration, dep_minutes, day_val, weather_val]]

# Predict delay
predicted_delay = model.predict(input_features)[0]

print("\n🚦 Predicted Delay:", round(predicted_delay, 2), "minutes")

# Suggestion based on threshold
if predicted_delay < 10:
    print("On-time or minimal delay expected.")
elif predicted_delay < 30:
    print("Moderate delay expected. Consider leaving earlier.")
else:
    print("Significant delay expected. You may want to reschedule.")
