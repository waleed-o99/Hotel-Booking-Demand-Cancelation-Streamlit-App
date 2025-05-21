%%writefile app.py
import streamlit as st
import pandas as pd
import os
import joblib
from datetime import date, timedelta

# Load model and scaler
model = joblib.load("best_lightgbm_model.pkl")
CSV_FILE = "predict_data.csv"
data_input = "saved_data.csv"

column=['hotel', 'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'babies', 'meal', 'country', 'distribution_channel',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'deposit_type', 'days_in_waiting_list', 'customer_type', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests']

# Create CSV if doesn't exist
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=column).to_csv(CSV_FILE, index=False)

if not os.path.exists(data_input):
    pd.DataFrame(columns=column).to_csv(data_input, index=False)

# App title and sidebar navigation
st.set_page_config(page_title="Hotel Booking Cancelation Prediction App", layout="centered")
st.title("🏨 Hotel Booking Prediction")
tabs = st.tabs(["📥 Input & Save Data", "🔍 Predict Saved Data"])

# Tab 1: Input and Save
with tabs[0]:
    st.header("📝 Enter Data")
    f1= st.selectbox("Hotel Type", options=["City Hotel", "Resort Hotel"])

# input duration to calculate lead_time, stays_in_weekend_nights, stays_in_week_nights
    today = date.today()
    d = st.date_input(
        "Select Reservation Dates", (today, ))
    if len(d)==1:
      f2 = None
      f3 = None
      f4 = None
    if len(d) == 2:
      # Create two datetime objects
      date1 = d[0]
      date2 = d[1]
      if date1 > date2 or date1 < today:
        st.warning("⚠️ Please select a valid date range.")
        f2 = None
        f3 = None
        f4 = None
      else:
        # Calculate the number of days between the two dates
        delta = date1 - today
        lead = delta.days
        weekend_nights_count = 0
        week_nights_count = 0
        current_date = date1 = d[0]
        while current_date <= date2:
            if current_date.weekday() >= 5:  # Saturday is 5, Sunday is 6
                weekend_nights_count += 1
            else:
                week_nights_count += 1
            current_date += timedelta(days=1)
        f2 = lead
        f3 = weekend_nights_count
        f4 = week_nights_count

    f5 = st.number_input("Adults", value=0)
    f6 = st.number_input("Children", value=0)
    f7 = st.number_input("Babies", value=0)
    f8 = st.multiselect("Meal", options=["Breakfast", "Lunch", "Dinner"])
    f9 = st.number_input("Country", value=0) ######
    f10 = st.selectbox("Distribution Channel", options=[ "Direct", "Corporate", "TA/TO", "GDS"])
    f11 = st.checkbox("Is Repeated Guest")
    f12 = st.number_input("Previous Cancellations", value=0)
    f13 = st.number_input("Previous Bookings Not Canceled", value=0)
    f14 = st.selectbox("Deposit Type", options=["No Deposit", "Refundable", "Non Refund"])
    f15 = st.number_input("Days in Waiting List", value=0) ############
    f16 = st.selectbox("Customer Type", options=["Transient","Contract", "Transient-Party", "Group"])
    f17 = st.number_input("Average Daily Rate", value=0) ######
    f18 = st.number_input("Car Parking Spaces", value=0)
    f19 = st.selectbox("Total of Special Requests", options=[0,1,2,3,4,5])

    row_input = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, 
                                f13, f14, f15, f16, f17, f18, f19]], columns=column)

#___________________is_repeated_guest_____________________________
    if f11:
      f11= 1
    else:
      f11= 0
#__________________Hotel__________________________________
    hotel = { "City Hotel": 1, "Resort Hotel": 0 }
    for key, value in hotel.items():
        if key == f1:
            f1 = value
#______________________Meal___________________________________________
    if len(f8) == None:
        f8 = 0
    else:
        f8 = len(f8)
#_______________________distribution_channel____________________________

    distribution_channel = { "Direct": 0, "Corporate": 1, "TA/TO": 2, "GDS": 3 }
    for key, value in distribution_channel.items():
        if key == f10:
            f10 = value
#__________________________Deposit_Type_______________________________

    deposit_type = { "No Deposit": 0, "Refundable": 1, "Non Refund": 2 }
    for key, value in deposit_type.items():
        if key == f14:
            f14 = value
#__________________________customer_type_______________________________

    customer_type = { "Transient": 0, "Contract": 1, "Transient-Party": 2, "Group": 3 }
    for key, value in customer_type.items():
        if key == f16:
            f16 = value
#___________________________________________________________________________

    if st.button("💾 Save Input"):
        new_row = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, 
                                 f13, f14, f15, f16, f17, f18, f19]], columns=column)

        if (new_row.isna().sum() > 0).any():
            st.warning("⚠️ Please fill in all required fields:")
            null_columns = new_row.columns[new_row.isnull().any()]
            if f2 == None:
                null_columns = null_columns.drop(["lead_time", "stays_in_weekend_nights", "stays_in_week_nights"])
                st.warning(" reservation dates is Required")

            #print please fill null_columns
            #st.warning("⚠️ Please fill in all required fields.")
            for i in null_columns:
              st.warning(f"{i} is Required")
          
        else:
            data = pd.read_csv(data_input)
            data = pd.concat([data, row_input], ignore_index=True)
            data.to_csv(data_input, index=False)
            
            #Data as Num. for predict model
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
            preds_df = pd.DataFrame(preds, columns=['Prediction'])
            pred_proba = model.predict_proba(df)
            prob_df = pd.DataFrame(pred_proba, columns=['Not Cancel Prob','Cancel Prob'])
            pr = pd.concat([preds_df, prob_df], axis=1)

            st.dataframe(data = pd.read_csv(data_input))
            st.dataframe(pr)
            st.success("✅ Prediction completed!")
            st.balloons()

