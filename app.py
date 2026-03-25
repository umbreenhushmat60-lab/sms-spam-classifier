import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# Define text transformation function
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("email/sms spam classifier")
input_sms = (st.text_area(" enter the message"))
if st.button('predict'):


    #1. preprocess
    transformed_text = transform_text(input_sms)
    #2. vectorize
    vector_input = tfidf.transform([transformed_text])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. display
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")

