# =============================================
# 🚌 Daewoo Bus Delay Predictor (Streamlit App)
# =============================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import datetime as dt
import requests

# -------------------------------
# 1️⃣ Load and Prepare Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_daewoo_delay_data.csv")

    # Encode categorical variables
    encoders = {}
    for col in ["Origin", "Destination", "Day", "Weather"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Convert departure time to minutes
    df["Departure"] = df["Departure"].apply(lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1]))
    return df, encoders

df, encoders = load_data()

# -------------------------------
# 2️⃣ Train the Model
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df[["Origin", "Destination", "Scheduled_Duration(min)", "Departure", "Day", "Weather"]]
    y = df["Delay(min)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mae, r2

model, mae, r2 = train_model(df)

# -------------------------------
# 3️⃣ Streamlit UI
# -------------------------------
st.title("🚌 Daewoo Bus Delay Predictor")
st.markdown("Predict expected delay based on route, time, and weather.")

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Mean Absolute Error", f"{mae:.2f} min")
st.sidebar.metric("R² Score", f"{r2:.2f}")

# -------------------------------
# 4️⃣ User Input Form
# -------------------------------
st.subheader("Enter Trip Details")

col1, col2 = st.columns(2)
with col1:
    origin_input = st.text_input("Origin City", "Lahore").title()
with col2:
    destination_input = st.text_input("Destination City", "Islamabad").title()

col3, col4 = st.columns(2)
with col3:
    date_input = st.date_input("Travel Date", dt.date.today())
with col4:
    time_input = st.time_input("Departure Time", dt.time(9, 0))

# -------------------------------
# 🌦️ Weather Detection Function
# -------------------------------
def get_weather_condition(city, date):
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
        return "Clear"

    lat, lon = city_coords[city]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weathercode&start_date={date}&end_date={date}&timezone=auto"
    response = requests.get(url)
    data = response.json()

    try:
        code = data["daily"]["weathercode"][0]
    except (KeyError, IndexError):
        return "Clear"

    if code in [0, 1]:
        return "Clear"
    elif code in [2, 3, 45, 48]:
        return "Foggy"
    elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
        return "Rainy"
    else:
        return "Clear"

# -------------------------------
# 5️⃣ Prediction Logic
# -------------------------------
if st.button("🔮 Predict Delay"):
    weather_input = get_weather_condition(origin_input, str(date_input))
    st.info(f"🌦️ Weather detected for **{origin_input}** on {date_input}: **{weather_input}**")

    # Extract weekday name
    day_name = date_input.strftime("%A")

    # Safe encoding with fallback
    def safe_encode(encoder, value):
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0

    origin_val = safe_encode(encoders["Origin"], origin_input)
    dest_val = safe_encode(encoders["Destination"], destination_input)
    day_val = safe_encode(encoders["Day"], day_name)
    weather_val = safe_encode(encoders["Weather"], weather_input)

    dep_minutes = time_input.hour * 60 + time_input.minute
    avg_duration = df["Scheduled_Duration(min)"].mean()

    features = [[origin_val, dest_val, avg_duration, dep_minutes, day_val, weather_val]]
    predicted_delay = model.predict(features)[0]

    st.subheader(f"🚦 Predicted Delay: {predicted_delay:.2f} minutes")

    if predicted_delay < 10:
        st.success("✅ On-time or minimal delay expected.")
    elif predicted_delay < 30:
        st.warning("⚠️ Moderate delay expected. Consider leaving earlier.")
    else:
        st.error("❌ Significant delay expected. You may want to reschedule.")
