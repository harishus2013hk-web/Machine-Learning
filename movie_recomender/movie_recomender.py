import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv(r'D:\Python\ML\Movie Recomender\tmdb_5000_movies.csv')
credits = pd.read_csv(r'D:\Python\ML\Movie Recomender\tmdb_5000_credits.csv')

# Merge datasets
df = movies.merge(credits, on='title')

# Required columns
df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Remove null values
df.dropna(inplace=True)

# Convert string list to actual list
def convert(text):
    return [i['name'].replace(" ", "") for i in ast.literal_eval(text)]

# Fetch top 3 cast
def convert_cast(text):
    return [i['name'].replace(" ", "") for i in ast.literal_eval(text)[:3]]

# Fetch director
def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name'].replace(" ", "")]
    return []

# Apply transformations
df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)
df['cast'] = df['cast'].apply(convert_cast)
df['crew'] = df['crew'].apply(fetch_director)
df['overview'] = df['overview'].apply(lambda x: x.split())

# Create tags
df['tags'] = (
    df['overview'] +
    df['genres'] +
    df['keywords'] +
    df['cast'] +
    df['crew']
).apply(lambda x: " ".join(x).lower())

# Stemming
ps = PorterStemmer()

df['tags'] = df['tags'].apply(
    lambda x: " ".join([ps.stem(word) for word in x.split()])
)

# Vectorization
tv = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tv.fit_transform(df['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    idx = df[df['title'] == movie].index[0]

    distances = sorted(
        list(enumerate(similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nRecommended movies for {movie}:\n")

    for i in distances:
        print(df.iloc[i[0]].title)

# Test
recommend("El Mariachi")