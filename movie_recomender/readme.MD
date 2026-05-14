# 🎬 Movie Recommendation System

A Content-Based Movie Recommendation System built using Python and Machine Learning techniques. The project recommends similar movies based on movie metadata such as genres, keywords, cast, crew, and overview.

---

## 🚀 Features

- Content-Based Movie Recommendation
- NLP Text Processing
- Cosine Similarity Matching
- Movie Metadata Analysis
- Top 5 Similar Movie Suggestions
- Beginner-Friendly Machine Learning Project

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK

---

## 📂 Dataset

Dataset used:
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

Dataset Source:
TMDB (The Movie Database)

---

## 📌 Project Workflow

1. Load datasets
2. Merge movie and credits data
3. Data preprocessing and cleaning
4. Extract important features:
   - Genres
   - Keywords
   - Cast
   - Director
   - Overview
5. Create combined tags
6. Apply stemming using NLTK
7. Convert text into vectors using CountVectorizer / TF-IDF
8. Calculate cosine similarity
9. Recommend similar movies

---

## 🧠 Recommendation Logic

The recommendation system uses cosine similarity to find movies with similar content.

Movies are recommended based on:
- Genres
- Keywords
- Cast
- Director
- Movie Overview

---

## 📦 Required Libraries

Install dependencies using:

```bash
pip install pandas numpy scikit-learn nltk
