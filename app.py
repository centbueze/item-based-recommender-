from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import joblib
import os

app = Flask(__name__)

# Load pre-trained models and data
def load_models():
    tfidf = joblib.load('data/trained_models/tfidf_vectorizer.pkl')
    cosine_sim = joblib.load('data/trained_models/cosine_similarity_matrix.pkl')
    indices = joblib.load('data/trained_models/indices.pkl')
    df = pd.read_csv('data/netflix_titles.csv', encoding='latin-1').fillna('')
    return tfidf, cosine_sim, indices, df

tfidf, cosine_sim, indices, df = load_models()

# Fuzzy title matching
def fuzzy_match_title(query, titles, threshold=70):
    match = process.extractOne(query, titles)
    return match[0] if match and match[1] >= threshold else None

# Recommendation logic
def recommend(title, top_n=5):
    matched_title = fuzzy_match_title(title, df['title'])
    if not matched_title:
        return None
    
    idx = indices.get(matched_title)
    if idx is None:
        return None
    
    sim_scores = sorted(
        enumerate(cosine_sim[idx]), 
        key=lambda x: x[1], 
        reverse=True
    )[1:top_n + 1]
    
    results = []
    for i, score in sim_scores:
        entry = df.iloc[i][['title', 'type', 'listed_in']]
        results.append({
            'title': entry['title'],
            'similarity': round(score, 3),
            'type': entry['type'],
            'genre': entry['listed_in']
        })
    return results

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    title = request.form.get('title')
    if not title:
        return jsonify({'error': 'No title provided'}), 400
    
    recommendations = recommend(title)
    if not recommendations:
        return jsonify({'error': f'Title "{title}" not found'}), 404
    
    return jsonify(recommendations)


@app.route("/about/")
def about():
     return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)