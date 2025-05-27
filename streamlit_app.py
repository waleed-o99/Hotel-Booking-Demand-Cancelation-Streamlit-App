import base64
import streamlit as st
import pandas as pd
import os
import joblib
from datetime import date, timedelta

def sidebar_bg(side_bg):

   side_bg_ext = 'png'
   st.markdown(
      f"""
      <style>
      [data-testid="stAppViewContainer"]  {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-size: cover;
          background-attachment: local;
      }}
      [data-testid="stWidgetLabel"]{{
        background-color: #F0F2F6;
      }}

      [data-testid="stToolbar"] {{
      right: 2rem;
      }}

      .stTabs [data-baseweb="tab"] {{
		      height: 50px;
          white-space: pre-wrap;
		      background-color: #F0F2F6;
		      border-radius: 14px 14px 0px 0px;
		      gap: 1px;
		      padding-top: 10px;
		      padding-bottom: 10px;
      }}

      .stTabs [aria-selected="true"] {{
        background-color: #b6fac4;
      }}

      [data-testid="stElementContainer"]{{
        background-color: #F0F2F6;
      }}

      [data-testid="stMarkdownContainer"]{{
          white-space: pre-wrap;
		      border-radius: 4px 4px 0px 0px;
		      gap: 1px;
		      padding-left: 20px;
          padding-right: 20px;
          
      }}
      [data-testid="stHeadingWithActionElements"]{{

          height: 90px;
        
      }}
      
      [data-testid="stButton"]{{
          padding-left: 50px;
          
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
 #          padding-bottom: 100px;  
#padding-bottom: 10px;
#height: 50px;

   
side_bg = 'hotel_img.jpg'
#side_bg = '/content/hotel_img1.jpg'
#side_bg = '/content/hotel_img2.jpg'
sidebar_bg(side_bg)




# Load model and scaler
model = joblib.load("best_lightgbm_model.pkl")
CSV_FILE = "predict_data.csv"
data_input = "saved_data.csv"

column=['hotel', 'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'babies', 'meal', 'country', 'distribution_channel',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'deposit_type', 'days_in_waiting_list', 'customer_type', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests']

country_map = {'Albania': 3, 'Algeria': 48, 'Andorra': 4, 'Angola': 1, 'Argentina': 6, 'Armenia': 7,
 'Australia': 11, 'Austria': 12, 'Azerbaijan': 13, 'Bahrain': 20, 'Bangladesh': 18, 'Belarus': 23,
 'Belgium': 15, 'Bolivia': 24, 'Bosnia and Herzegovina': 22, 'Brazil': 25, 'Bulgaria': 19,
 'Cabo Verde': 37, 'Cameroon': 33, 'Central African Republic': 28, 'Chile': 30, 'China': 31, 'Colombia': 35,
 'Costa Rica': 38, 'Croatia': 71, 'Cuba': 39, 'Cyprus': 41, 'Czechia': 42, "Côte d'Ivoire": 32,
 'Denmark': 46, 'Dominican Republic': 47, 'Ecuador': 49, 'Egypt': 50, 'Estonia': 52, 'Finland': 54,
 'France': 56, 'Gabon': 58, 'Georgia': 60, 'Germany': 43, 'Ghana': 62, 'Gibraltar': 63, 'Greece': 66,
 'Guinea-Bissau': 65, 'Hong Kong': 69, 'Hungary': 72, 'Iceland': 79, 'India': 75, 'Indonesia': 73,
 'Iran, Islamic Republic of': 77, 'Iraq': 78, 'Ireland': 76, 'Italy': 81, 'Jamaica': 82, 'Japan': 85,
 'Jersey': 83, 'Jordan': 84, 'Kazakhstan': 86, 'Kenya': 87, 'Korea, Republic of': 91, 'Kuwait': 92,
 'Latvia': 101, 'Lebanon': 94, 'Lithuania': 99, 'Luxembourg': 100, 'Macao': 102, 'Malaysia': 117,
 'Maldives': 106, 'Malta': 110, 'Mauritius': 115, 'Mexico': 107, 'Monaco': 104, 'Montenegro': 112,
 'Morocco': 103, 'Mozambique': 113, 'Netherlands': 123, 'New Zealand': 126, 'Nigeria': 121,
 'North Macedonia': 108, 'Norway': 124, 'Oman': 127, 'Pakistan': 128, 'Palestine': 80, 'Panama': 129,
 'Paraguay': 136, 'Peru': 130, 'Philippines': 131, 'Poland': 133, 'Portugal': 135, 'Puerto Rico': 134,
 'Qatar': 138, 'Romania': 139, 'Russian Federation': 140, 'Saudi Arabia': 142, 'Senegal': 144,
 'Serbia': 149, 'Singapore': 145, 'Slovakia': 152, 'Slovenia': 153, 'South Africa': 174,
 'Spain': 51, 'Sri Lanka': 98, 'Sweden': 154, 'Switzerland': 29, 'Syrian Arab Republic': 156,
 'Taiwan, Province of China': 163, 'Tanzania, United Republic of': 164, 'Thailand': 158,
 'Tunisia': 161, 'Turkey': 162, 'Ukraine': 166, 'United Arab Emirates': 5, 'United Kingdom': 59,
 'United States': 169, 'Uruguay': 168, 'Uzbekistan': 170, 'Venezuela, Bolivarian Republic of': 171,
 'Viet Nam': 173, 'Zimbabwe': 176}

country_names = list(country_map.keys())

# Create CSV if doesn't exist
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=column).to_csv(CSV_FILE, index=False)

if not os.path.exists(data_input):
    pd.DataFrame(columns=column).to_csv(data_input, index=False)

# App title and sidebar navigation
#st.set_page_config(page_title="Hotel Booking Cancelation Prediction App", layout="centered")

st.title("🏨 Hotel Booking Prediction")
tabs = st.tabs(["  📥 Input & Save Data  ", "  🔍 Predict Saved Data  "])

# Tab 1: Input and Save
with tabs[0]:
    st.header("📝 Enter Data")
    f1= st.selectbox("Hotel Type", options=["City Hotel", "Resort Hotel"], index=None, placeholder="Select an option...")

# input duration to calculate lead_time, stays_in_weekend_nights, stays_in_week_nights
    today = date.today()
    d = st.date_input("Reservation Dates", (today, ),help="select dates (start, end)")
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
#,step =1, label_visibility , help
    f5 = st.number_input("Adults", min_value=0, step =1)
    f6 = st.number_input("Children", min_value=0)
    f7 = st.number_input("Babies", min_value=0)
    f8 = st.multiselect("Meal", options=["Breakfast", "Lunch", "Dinner"], placeholder="Select all meal package needed...")
    
    f9 = st.selectbox("Country", options=country_names, index=None, placeholder="Select your Country...", help="Where are you come from?")
    
    
    f10 = st.selectbox("Distribution Channel", options=[ "Direct", "Corporate", "TA/TO", "GDS"], index=None,
                       placeholder="Select an option...", help="TA/TO:Travel Agents/Tour Operators.   \n GDS: Global Distribution System ")
    #f11 = st.checkbox("Is Repeated Guest")

    f12 = st.number_input("Previous Cancellations", min_value=0)
    f13 = st.number_input("Previous Bookings Not Canceled", min_value=0)
    f14 = st.selectbox("Deposit Type", options=["No Deposit", "Refundable", "Non Refund"], index=None,
                       placeholder="Select an option...", help= "Refundable: value under the total cost of stay.   \n Non Refund: value of the total stay cost.")
    f15 = st.number_input("Days in Waiting List", min_value=0) ############
    f16 = st.selectbox("Customer Type", options=["Transient","Contract", "Transient-Party", "Group"], index=None, placeholder="Select an option...")
    f17 = st.number_input("Average Daily Rate", min_value=0) ######
    f18 = st.number_input("Car Parking Spaces", min_value=0)
    f19 = st.selectbox("Total of Special Requests", options=[0,1,2,3,4,5], index=None, placeholder="Select an option...")

#______________________is_repeated_guest_____________________________
    # if there is a Previous Cancellations or not Canceled means repeated_guest
    if f12 + f13 == 0:
      f11 = 0
    else:
      f11 = 1
# Save input as DF________________________________________________________________
    row_input = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, 
                                f13, f14, f15, f16, f17, f18, f19]], columns=column)
#____________________________Hotel__________________________________
    hotel = { "City Hotel": 1, "Resort Hotel": 0 }
    for key, value in hotel.items():
        if key == f1:
            f1 = value
#______________________Meal___________________________________________
    if len(f8) == None:
        f8 = 0
    else:
        f8 = len(f8)
#__________________________country_______________________________
    for key, value in country_map.items():
        if key == f9:
            f9 = value
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
          
	elif f5 + f6 == 0
	    st.warning("Pleas enter number of guest")
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

