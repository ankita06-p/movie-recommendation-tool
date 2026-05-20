import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cast=pd.read_csv("cast.csv")
movies=pd.read_csv("movies.csv")
movies=movies[['title', 'genres', 'overview', 'keywords', 'spoken_languages']]

def fetch_director(obj):
    director = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            director.append(i['name'])
    return director

movies['director'] = cast['crew'].apply(fetch_director)
missing_values=movies[movies.isnull().any(axis=1)]
print(missing_values)

movies.dropna(inplace=True)
movies.reset_index(drop=True, inplace=True)
movies['tags'] = movies[['genres', 'overview', 'keywords', 'director']].astype(str).agg(' '.join, axis=1)
movies['tags'] = movies['tags'].str.replace(r"[\[\]',]", "", regex=True).str.lower()

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=5000, stop_words='english')
vectors= cv.fit_transform(movies['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity= cosine_similarity(vectors)

def recommendation(movie):
  movie_index= movies[movies['title']==movie].index[0]
  distances= similarity[movie_index]

  movie_list = list(enumerate(distances))
  movie_list = sorted(movie_list, key=lambda x: x[1], reverse=True)[1:6]

  for i in movie_list:
    print(movies.iloc[i[0]].title)

