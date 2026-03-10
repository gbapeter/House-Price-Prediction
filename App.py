import streamlit as st 
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Ensure datetime
        X['date'] = pd.to_datetime(X['date'])

        # Extract features
        X['month'] = X['date'].dt.month
        X['year'] = X['date'].dt.year

        return X

def log_func(x):
    return np.log1p(x)

def exp_func(x):
    return np.expm1(x)
model = joblib.load('group4.pkl')

# All cities from training data 
cities = [
    "Algona", "Auburn", "Bellevue", "Black Diamond", "Bothell", "Burien",
    "Carnation", "Clyde Hill", "Covington", "Des Moines", "Duvall",
    "Enumclaw", "Fall City", "Federal Way", "Issaquah", "Kenmore", "Kent",
    "Kirkland", "Lake Forest Park", "Maple Valley", "Medina", "Mercer Island",
    "Milton", "Newcastle", "Normandy Park", "North Bend", "Pacific", "Preston",
    "Ravensdale", "Redmond", "Renton", "Sammamish", "SeaTac", "Seattle",
    "Shoreline", "Skykomish", "Snoqualmie", "Snoqualmie Pass", "Tukwila",
    "Vashon", "Woodinville", "Yarrow Point"
]


st.title("HOUSE PRICE PREDICTION APPLICATION")
st.write("Predict the Price of a House Using a Random Forest Model")


form = st.form("house_price_form")
form.subheader("Enter House Details")

date = form.text_input("Sale Date (YYYY-MM-DD)", value="2014-05-01")
bedrooms = form.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = form.slider("Number of Bathrooms", min_value=1, max_value=8, value=2)
sqft_living = form.number_input("Living Area (sqft)", min_value=200, max_value=15000, value=1500)
sqft_lot = form.number_input("Lot Size (sqft)", min_value=500, max_value=150000, value=5000)
floors = form.slider("Number of Floors", min_value=1, max_value=4, value=1)
waterfront = form.selectbox("Waterfront Property?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
view = form.slider("View Quality (0-4)", min_value=0, max_value=4, value=0)
condition = form.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
sqft_above = form.number_input("Sqft Above Ground", min_value=200, max_value=10000, value=1500)
sqft_basement = form.number_input("Sqft Basement", min_value=0, max_value=5000, value=0)
yr_built = form.number_input("Year Built", min_value=1900, max_value=2024, value=2000)
yr_renovated = form.number_input("Year Renovated (0 if never)", min_value=0, max_value=2024, value=0)
street = form.text_input("Street", value="123 Main St")
city = form.selectbox("City", options=cities)
statezip = form.text_input("State Zip (e.g. WA 98001)", value="WA 98001")
country = form.text_input("Country", value="USA")

submit = form.form_submit_button("Predict Price")

if submit:
    try:
        input_data = pd.DataFrame({
            'date': [date],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'condition': [condition],
            'sqft_above': [sqft_above],
            'sqft_basement': [sqft_basement],
            'yr_built': [yr_built],
            'yr_renovated': [yr_renovated],
            'street': [street],
            'city': [city],
            'statezip': [statezip],
            'country': [country]
        })

        prediction = model.predict(input_data)[0]

        
        st.success(f"Estimated House Price: **${prediction:,.2f}**")

        with st.expander("View Input Summary"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"❌ Error: {e}")

        st.warning("Check that your group4.pkl is in the same folder as this script.")

