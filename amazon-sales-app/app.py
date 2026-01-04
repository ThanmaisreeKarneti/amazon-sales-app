import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
# ------------------ Load Dataset ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("amazon.csv", encoding="latin1")
    except Exception:
        df = pd.read_csv("amazon.csv", encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.lower().str.strip()
    # Convert numeric columns
    for col in ["discount_price", "actual_price", "discount_percentage", "rating", "rating_count"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")   
    # Compute price_diff
    if "actual_price" in df.columns and "discount_price" in df.columns:
        df["price_diff"] = df["actual_price"] - df["discount_price"]   
    return df
# ------------------ EDA ------------------
def run_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("### Dataset Overview")
    st.dataframe(df.head())
    # Average Discount by Category
    if "category" in df.columns and "discount_price" in df.columns:
        valid_df = df.dropna(subset=["category", "discount_price"])
        if not valid_df.empty:
            fig, ax = plt.subplots(figsize=(10, 2))
            sns.barplot(data=valid_df, x="category", y="discount_price", ci=None, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("Average Discount Price by Category")
            st.pyplot(fig)
    # Ratings distribution
    if "rating" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.histplot(df["rating"].dropna(), kde=True, color="orange", ax=ax)
        ax.set_title("Ratings Distribution")
        st.pyplot(fig)
 # Scatterplot of Discount Price vs Actual Price
    if "discount_price" in df.columns and "actual_price" in df.columns:
        valid_df = df.dropna(subset=["discount_price", "actual_price"])
        if not valid_df.empty:
            fig, ax = plt.subplots(figsize=(8, 2))
            sns.scatterplot(data=valid_df, x="actual_price", y="discount_price", ax=ax)
            ax.set_title("Discount Price vs Actual Price")
            st.pyplot(fig)
# ------------------ Dataset Metrics ------------------
def calculate_metrics(df):
    if "review_content" in df.columns:
        df["review_sentiment"] = df["review_content"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    else:
        df["review_sentiment"] = 0
    df = df.dropna(subset=["actual_price", "discount_price"])
    df["discount_percentage"] = ((df["actual_price"] - df["discount_price"]) / df["actual_price"]) * 100
    df["price_diff"] = df["actual_price"] - df["discount_price"]
    X_reg = df[["actual_price", "discount_percentage", "rating", "rating_count", "price_diff", "review_sentiment"]]
    y_reg = df["discount_price"]
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, reg_model.predict(X_test))*100
    r2 = r2_score(y_test, reg_model.predict(X_test))
    return mse, r2
# ------------------ Affordability & Discount Prediction ------------------
def run_prediction(df, actual_price, discount_per, rating, rating_count):
    # Affordability logic
    if rating < 4:
        affordability = "Not Affordable ⚠️"
    elif actual_price < 20000:
        affordability = "Affordable ✅"
    else:
        affordability = "Expensive 💰"
    # Price-based category lookup
    matched = df[df['actual_price'] == actual_price]
    if matched.empty:
        categories = "No category found with this price."
    else:
        categories = matched['category'].dropna().unique().tolist()
    # Regression prediction for input
    df = df.dropna(subset=["actual_price", "discount_price"])
    df["price_diff"] = df["actual_price"] - df["discount_price"]
    if "review_content" in df.columns:
        df["review_sentiment"] = df["review_content"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    else:
        df["review_sentiment"] = 0
    X_reg = df[["actual_price", "discount_percentage", "rating", "rating_count", "price_diff", "review_sentiment"]]
    y_reg = df["discount_price"]
    reg_model = LinearRegression()
    reg_model.fit(X_reg, y_reg)
    discounted_price_input = actual_price * (1 - discount_per/100)
    input_data_reg = np.array([[actual_price, discount_per, rating, rating_count, discounted_price_input, 0]])
    y_pred_input = reg_model.predict(input_data_reg)[0]
    return affordability, categories, y_pred_input
# ------------------ NLP ------------------
def run_nlp(df, predicted_categories=None):
    st.header("🧠 Sentiment Analysis on Reviews")
    df["sentiment_label"] = df["review_content"].astype(str).str.upper().replace({
        "GOOD": "Positive",
        "BAD": "Negative",
        "AVERAGE": "Neutral"
    })
    filtered_df = df[df["sentiment_label"].isin(["Positive", "Negative", "Neutral"])]
    # Graph 1
    st.subheader("1️⃣ Total Sentiment Count")
    sentiment_counts = filtered_df["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig, ax = plt.subplots()
    sns.barplot(data=sentiment_counts, x="Sentiment", y="Count", ax=ax)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    st.pyplot(fig)
    # Graph 2
    st.subheader("2️⃣ Sentiment Count Per Category")
    category_sentiment = (
        filtered_df.groupby(["category", "sentiment_label"])["review_content"]
        .count()
        .reset_index()
        .rename(columns={"review_content": "Count"})
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=category_sentiment, x="category", y="Count", hue="sentiment_label", ax=ax)
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    st.pyplot(fig)
    # Graph 3
    st.subheader("3️⃣ Sentiment Count for Predicted Category")
    if predicted_categories:
        if isinstance(predicted_categories, str):
            predicted_categories = [predicted_categories]
        filtered_pred = filtered_df[filtered_df["category"].isin(predicted_categories)]
        if not filtered_pred.empty:
            predicted_sentiment = (
                filtered_pred.groupby(["category", "sentiment_label"])["review_content"]
                .count()
                .reset_index()
                .rename(columns={"review_content": "Count"})
            )
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(data=predicted_sentiment, x="category", y="Count", hue="sentiment_label", ax=ax)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            st.pyplot(fig)
# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Amazon Dashboard", layout="wide")
st.title("📦 Amazon Sales Dashboard")
st.caption("EDA + Affordability Prediction + Sentiment Analysis")
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()
tab1, tab2, tab3 = st.tabs(["EDA", "Prediction", "NLP Analysis"])
with tab1:
    run_eda(df)
with tab2:
    # Show dataset metrics immediately
    mse, r2 = calculate_metrics(df)
    st.subheader("📊 Dataset Metrics (always shown)")
    st.write(f"MSE: {mse}")
    st.write(f"R2 Score: {r2}")
    # Input fields
    st.subheader("💰 Product Affordability & Discount Prediction")
    actual_price = st.number_input("Actual Price", min_value=0.0, value=199.0)
    discount_per = st.number_input("Discount %", min_value=0.0, value=5.0)
    rating = st.number_input("Rating (0-5)", min_value=0.0, max_value=5.0, step=0.1, value=4.5)
    rating_count = st.number_input("Rating Count", min_value=0.0, value=100.0)
    # Run prediction immediately using default input values
    affordability, categories, y_pred_input = run_prediction(
        df, actual_price, discount_per, rating, rating_count
    )
    # Predict button
    if st.button("Predict"):
        affordability, categories, y_pred_input = run_prediction(
            df, actual_price, discount_per, rating, rating_count
        )
    # Display prediction results
    st.subheader("💰 Prediction Results ")
    st.write(f"Affordability: {affordability}")
    st.write(f"Categories: {categories}")
    st.write(f"Predicted Discounted Price: ₹{y_pred_input:.2f}")
    predicted_category = categories  # from run_prediction
with tab3:
    run_nlp(df, predicted_categories=predicted_category)

