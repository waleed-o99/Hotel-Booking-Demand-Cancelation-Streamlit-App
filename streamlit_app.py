import streamlit as st
import pandas as pd
import os
import joblib

# Load model and scaler
model = joblib.load("best_xgboost_model.pkl")
CSV_FILE = "saved_data.csv"

column=['hotel', 'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
                        'adults', 'children', 'babies', 'meal', 'country', 'distribution_channel',
                        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
                        'deposit_type', 'days_in_waiting_list', 'customer_type', 'adr',
                        'required_car_parking_spaces', 'total_of_special_requests']

# Create CSV if doesn't exist
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=column).to_csv(CSV_FILE, index=False)

# App title and sidebar navigation
st.set_page_config(page_title="Hotel Booking Prediction App", layout="centered")
st.title("🏨 Hotel Booking Prediction")
tabs = st.tabs(["📥 Input & Save Data", "🔍 Predict Saved Data"])

# Tab 1: Input and Save
with tabs[0]:
    st.header("📝 Enter Data")
    f1 = st.number_input("Feature 1", value=0)
    f2 = st.number_input("Feature 2", value=0)
    f3 = st.number_input("Feature 3", value=0)
    f4 = st.number_input("Feature 4", value=0)
    f5 = st.number_input("Feature 5", value=0)
    f6 = st.number_input("Feature 6", value=0)
    f7 = st.number_input("Feature 7", value=0)
    f8 = st.number_input("Feature 8", value=0)
    f9 = st.number_input("Feature 9", value=0)
    f10 = st.number_input("Feature 10", value=0)
    f11 = st.number_input("Feature 11", value=0)
    f12 = st.number_input("Feature 12", value=0)
    f13 = st.number_input("Feature 13", value=0)
    f14 = st.number_input("Feature 14", value=0)
    f15 = st.number_input("Feature 15", value=0)
    f16 = st.number_input("Feature 16", value=0)
    f17 = st.number_input("Feature 17", value=0)
    f18 = st.number_input("Feature 18", value=0)
    f19 = st.number_input("Feature 19", value=0)
    
    if st.button("💾 Save Input"):
        new_row = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]], columns=column)
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        st.success("✅ Data saved successfully!")

# Tab 2: Predict from Saved Data
with tabs[1]:
    st.header("🤖 Prediction Results")
    if st.button("🔄 Predict Saved Data"):
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            st.warning("⚠️ No data available for prediction.")
        else:
            preds = model.predict(df)
            df["Prediction"] = preds
            st.dataframe(df)
