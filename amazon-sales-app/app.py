import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

# ----------------------------------------------------
# Streamlit Page Config
# ----------------------------------------------------
st.set_page_config(page_title="Amazon Sales Prediction", layout="wide")
st.title("📦 Amazon Sales Prediction App")

# ----------------------------------------------------
# File Upload
# ----------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload your Amazon CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------------------------------
    # Data Cleaning
    # ----------------------------------------------------
    cols_to_convert = ['discounted_price', 'actual_price',
                       'discount_percentage', 'rating', 'rating_count']

    def clean_to_int(series):
        return pd.to_numeric(
            series.astype(str)
                  .str.replace(',', '', regex=False)
                  .str.replace('%', '', regex=False)
                  .str.replace('₹', '', regex=False)
                  .str.replace('[^0-9.\-]', '', regex=True),
            errors='coerce'
        )

    df[cols_to_convert] = df[cols_to_convert].apply(clean_to_int)
    df.dropna(subset=cols_to_convert, inplace=True)

    # New features
    df['price_difference'] = df['actual_price'] - df['discounted_price']
    df['discount_price'] = df['discounted_price'].apply(lambda x: 1 if x >= 500 else 0)

    # ----------------------------------------------------
    # Model Training
    # ----------------------------------------------------
    X = df[['actual_price', 'discount_percentage', 'rating',
            'rating_count', 'price_difference']]
    y = df['discount_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    st.success("✅ Model trained successfully on uploaded data!")

    # ----------------------------------------------------
    # Prediction Section
    # ----------------------------------------------------
    st.sidebar.header("🔮 Make a Prediction")
    actual_price = st.sidebar.number_input("Actual Price", min_value=0.0, value=1000.0)
    discount_per = st.sidebar.number_input("Discount %", min_value=0.0, value=10.0)
    rating = st.sidebar.number_input("Rating (0-5)", min_value=0.0, max_value=5.0, step=0.1, value=4.0)
    rating_count = st.sidebar.number_input("Rating Count", min_value=0.0, value=100.0)

    price_diff = actual_price - (actual_price * (1 - discount_per / 100))

    if st.sidebar.button("Predict"):
        input_data = np.array([[actual_price, discount_per, rating, rating_count, price_diff]])
        pred = clf.predict(input_data)[0]
        if pred == 1:
            st.error("Prediction: Expensive Product 💰")
        else:
            st.success("Prediction: Affordable Product ✅")

    # ----------------------------------------------------
    # Sentiment Analysis (if reviews exist)
    # ----------------------------------------------------
    if "review_content" in df.columns:
        st.subheader("📝 Sentiment Analysis on Reviews")
        df['review_sentiment'] = df['review_content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment_label'] = df['review_sentiment'].apply(
            lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
        )
        st.write(df[['review_content', 'sentiment_label']].head(10))
