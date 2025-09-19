# Task2/nlp_pipeline.py
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import pandas as pd
import re

# Make sure to download these once:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()
SIA = SentimentIntensityAnalyzer()

def clean_text(s):
    s = s.lower()
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    lemmas = [LEM.lemmatize(t) for t in tokens]
    pos = pos_tag(lemmas)
    return lemmas, pos

def dataframe_from_texts(texts):
    rows = []
    for i, t in enumerate(texts):
        lemmas, pos = preprocess_text(t)
        sscore = SIA.polarity_scores(t)
        rows.append({
            'id': i,
            'original': t,
            'tokens': lemmas,
            'pos': pos,
            'neg': sscore['neg'],
            'neu': sscore['neu'],
            'pos_score': sscore['pos'],
            'compound': sscore['compound']
        })
    return pd.DataFrame(rows)

def top_ngrams(texts, n=20, ngram_range=(1,2)):
    # very simple: flatten tokens and count
    counts = Counter()
    for t in texts:
        lemmas, _ = preprocess_text(t)
        for k in range(ngram_range[0], ngram_range[1]+1):
            for i in range(len(lemmas)-k+1):
                ng = "_".join(lemmas[i:i+k])
                counts[ng] += 1
    return counts.most_common(n)
