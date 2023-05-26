from django.shortcuts import render
import pickle
from django.db.models import Avg
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from spellchecker import SpellChecker
import nltk

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

clean_data = pd.read_csv('myapp/static/clean_data.csv', encoding='utf-8')

titles = clean_data["Book-Title"]
authors = clean_data["Book-Author"]
desc = clean_data["description"]
image_urls = clean_data["Image-URL-L"]

# Charger le fichier vectorizer.pkl
with open('myapp/static/vectorizer.pkl', 'rb') as fichier:
    vectorizer = pickle.load(fichier)

# Charger le fichier nbrs.pkl
with open('myapp/static/model.pkl', 'rb') as fichier:
    nbrs = pickle.load(fichier)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(w) for w in words]
    return ' '.join(words)

def correct_query(query):
    spell = SpellChecker()
    words = query.split()
    corrected_words = []
    for word in words:
        if not spell.correction(word) == word:
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    corrected_query = ' '.join(corrected_words)
    return corrected_query

def search_books(query):
    query = correct_query(query)
    query = clean_text(query)
    query_vector = vectorizer.transform([query])
    distances, indices = nbrs.kneighbors(query_vector)

    results = [(image_urls[idx], titles[idx], authors[idx], desc[idx]) for idx in indices[0]]
    return results


def home(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        results = search_books(query)
        return render(request, 'home.html', {'results': results, 'query': query})
    else:
        return render(request, 'home.html')


def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        results = search_books(query)
        request.session['results'] = results
    results = request.session.get('results', [])
    context = {'results': results}
    return render(request, 'search.html', context)