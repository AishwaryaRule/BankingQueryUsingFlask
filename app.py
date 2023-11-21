from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import nltk
import re
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

 
@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        input_question=request.form['question']

        stemmer=PorterStemmer()

        sentence=input_question.lower()                            #converting to lower case
        sentence =re.sub(r'[^\w\s]', '', sentence)              #removing punctuations,links,special characters
        sentence = re.sub(r"https\S+|www\S+https\S+", '',sentence, flags=re.MULTILINE)
        sentence = re.sub(r'\@w+|\#','',sentence)
        words=word_tokenize(sentence)                              #tokenization
        words=[stemmer.stem(word) for word in words if word not in stopwords.words("english")]           #removing stop-words and stemming
        processed=" ".join(words)

        vectorizer1=pickle.load(open('Vectorizer.pkl','rb'))
        question_vectors=pickle.load(open('Question_vectors.pkl','rb'))
        df=pickle.load(open('BANK_data.pkl','rb'))

        test_vector=vectorizer1.transform([processed]).toarray()

        cosine_sim=cosine_similarity(question_vectors,test_vector)
        most_sim_idx=np.argmax(cosine_sim)

        final_output=df.iloc[most_sim_idx]['Answer']

        
        return render_template('result.html',answer=final_output)



if __name__=='__main__':
    app.run(debug=True)