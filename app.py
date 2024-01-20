import pickle
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
import streamlit as st

st.title('SMS Spam Classifier')
content = st.text_area(label='Nội dung tin nhắn', placeholder='Nhập nội dung tin nhắn tại đây', height=200)
btn_verify = st.button('Phân loại')

vectorizer = pickle.load(open('./vectorizer.pkl', 'rb'))
log_reg_pretrain = pickle.load(open('./LogisticRegression.pkl', 'rb'))
svm_pretrain = pickle.load(open('./SupportVectorClassifier.pkl', 'rb'))

def transform_text(content):
    text = content.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(lemma.lemmatize(i))
    return " ".join(y)

lemma = WordNetLemmatizer()

def spam_predictor(content):
    clean_content = transform_text(content)
    input_data = [clean_content]
    vector_form_raw = vectorizer.transform(input_data)

    log_reg_pred = log_reg_pretrain.predict(vector_form_raw)
    log_reg_pred_proba = log_reg_pretrain.predict_proba(vector_form_raw)

    svm_pred = svm_pretrain.predict(vector_form_raw)
    svm_pred_proba = svm_pretrain.predict_proba(vector_form_raw)

    return log_reg_pred, log_reg_pred_proba, svm_pred, svm_pred_proba

if btn_verify:
    if content != '':
        log_reg_pred, log_reg_pred_proba, svm_pred, svm_pred_proba = spam_predictor(content)

        st.progress(value=round(((log_reg_pred_proba[0][1] + svm_pred_proba[0][1])/2), 2),
                    text=f'Spam probability: {round(((log_reg_pred_proba[0][1] + svm_pred_proba[0][1])/2), 3)}')
        if round(((log_reg_pred_proba[0][1] + svm_pred_proba[0][1])/2), 2) > 0.5:
            st.header("SPAM")
        else:
            st.header("Non-spam")
    else:
        st.warning('Thêm tin tức và nhấn "Xác thực" để anh giúp em kiểm tra xem tin này có spam hay không nhé!')
