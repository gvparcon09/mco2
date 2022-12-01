import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


def find_cosine_sim(df):

	#Replace NaN with an empty string
	df['overview'] = df['overview'].fillna('')

	#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
	tfidf = TfidfVectorizer(stop_words='english')

	#Construct the required TF-IDF matrix by fitting and transforming the data
	tfidf_matrix = tfidf.fit_transform(df['overview'])

	# Compute the cosine similarity matrix
	cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

	return cosine_sim


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim):
	# Get the index of the movie that matches the title
	idx = indices[title]

	# Get the pairwsie similarity scores of all movies with that movie
	sim_scores = list(enumerate(cosine_sim[idx]))

	# Sort the movies based on the similarity scores
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

	# Get the scores of the 10 most similar movies
	sim_scores = sim_scores[1:11]

	# Get the movie indices
	movie_indices = [i[0] for i in sim_scores]

	# Return the top 10 most similar movies
	return df['title'].iloc[movie_indices]


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
	for i in x:
		if i['job'] == 'Director':
			return i['name']
	return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
	if isinstance(x, list):
		names = [i['name'] for i in x]
		#Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
		if len(names) > 3:
			names = names[:3]
		return names

	#Return empty list in case of missing/malformed data
	return []

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
	if isinstance(x, list):
		return [str.lower(i.replace(" ", "")) for i in x]
	else:
		#Check if director exists. If not, return empty string
		if isinstance(x, str):
			return str.lower(x.replace(" ", ""))
		else:
			return ''

def create_soup(x):
	return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


if __name__== "__main__":

	#loading datasets
	df1=pd.read_csv('tmdb_5000_credits.csv')
	df2=pd.read_csv('tmdb_5000_movies.csv')
	#df=pd.read_csv('movies_metadata.csv')

	df1.columns = ['id','tittle','cast','crew']
	df= df2.merge(df1,on='id')
	
	
	print(df['overview'].head(5))

	cosine_sim = find_cosine_sim(df)

	#Construct a reverse map of indices and movie titles
	indices = pd.Series(df.index, index=df['title']).drop_duplicates()

	movie_rec = get_recommendations('The Dark Knight Rises',cosine_sim)
	print(movie_rec)

	print("===========================================================")
	
	features = ['cast', 'crew', 'keywords', 'genres']
	for feature in features:
		df[feature] = df[feature].apply(literal_eval)

	df['director'] = df['crew'].apply(get_director)

	features = ['cast', 'keywords', 'genres']
	for feature in features:
		df[feature] = df[feature].apply(get_list)

	# Apply clean_data function to your features.
	features = ['cast', 'keywords', 'director', 'genres']

	for feature in features:
		df[feature] = df[feature].apply(clean_data)

	df['soup'] = df.apply(create_soup, axis=1)

	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(df['soup'])

	cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

	# Reset index of our main DataFrame and construct reverse mapping as before
	df = df.reset_index()
	indices = pd.Series(df.index, index=df['title'])

	movie_rec = get_recommendations('The Dark Knight Rises',cosine_sim2)
	print(movie_rec)