# Task2/streamlit_app.py
import streamlit as st
import pandas as pd
from Task2.nlp_pipeline import dataframe_from_texts, top_ngrams
import matplotlib.pyplot as plt

st.set_page_config(page_title="NLTK Text Analytics", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Explorer", "Analysis Dashboard"])

uploaded = st.sidebar.file_uploader("Upload a .txt or .csv (one document per line)", type=['txt','csv'])

if uploaded:
    try:
        if uploaded.type == "text/csv" or uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
            # if single column, try first column
            if df.shape[1] == 1:
                texts = df.iloc[:,0].astype(str).tolist()
            else:
                # try to find a 'text' column
                if 'text' in df.columns:
                    texts = df['text'].astype(str).tolist()
                else:
                    st.warning("CSV has multiple columns; using first column as text")
                    texts = df.iloc[:,0].astype(str).tolist()
        else:
            # txt: one document per line
            texts = uploaded.read().decode('utf-8').splitlines()
    except Exception as e:
        st.error(f"Failed to read upload: {e}")
        texts = []
else:
    st.info("No upload — using example dataset")
    # load local example if exists
    try:
        df_ex = pd.read_csv("Task2/example_dataset.csv")
        texts = df_ex['text'].astype(str).tolist()
    except Exception:
        texts = ["This movie was great!", "I disliked the taste of that dish.", "Neutral comment here."]

if page == "Data Explorer":
    st.header("Data Explorer")
    st.write(f"Loaded {len(texts)} documents.")
    if st.checkbox("Show raw documents"):
        for i,t in enumerate(texts[:100]):
            st.write(f"{i}: {t}")

    if st.button("Build dataframe"):
        df = dataframe_from_texts(texts)
        st.dataframe(df.head(200))

if page == "Analysis Dashboard":
    st.header("Analysis Dashboard")
    with st.spinner("Computing top n-grams and sentiment..."):
        topn = st.sidebar.slider("Top N", 5, 100, 20)
        ngrams = top_ngrams(texts, n=topn, ngram_range=(1,2))
        st.subheader("Top n-grams")
        ng_df = pd.DataFrame(ngrams, columns=['ngram','count'])
        st.table(ng_df.head(50))

        # sentiment trend
        df = dataframe_from_texts(texts)
        st.subheader("Sentiment distribution (compound score)")
        fig, ax = plt.subplots()
        ax.hist(df['compound'], bins=30)
        ax.set_xlabel('compound'); ax.set_ylabel('count')
        st.pyplot(fig)

        if st.checkbox("Show POS tags sample"):
            sample = df.sample(min(5, len(df)))
            for _, row in sample.iterrows():
                st.write(row['original'])
                st.write(row['pos'])
