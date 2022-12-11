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
	sim_scores = sim_scores[1:21]

	print("sim score : ",sim_scores)

	# Get the movie indices
	movie_indices = [i[0] for i in sim_scores]

	# Return the top 10 most similar movies
	return ml_df['title'].iloc[movie_indices]

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

def clean_genre_data(x):
	return str.lower(str(x).replace("|", " "))

def clean_cast_data(x):
	return str.lower(str(x).replace("|", ""))


def split_title(x):
	movie_title = x
	year_start = x.find('(')
	if(year_start != -1):
		movie_year = x[year_start:year_start+4]
		movie_title = x[:year_start-1]
	return movie_title

def get_year(x):
	movie_year = None
	year_start = x.find('(')
	if(year_start != -1):
		movie_year = x[year_start+1:year_start+5]
	return movie_year

def create_soup(x):
	ret_val = str(x['genres']) + ' ' + str(x['casts']) + ' ' + str(x['director'])
	return ret_val

if __name__== "__main__":

	#loading datasets
	ml_df=pd.read_csv('movies-director-casts.csv')	# genre/director/cast metadata
	og_df=pd.read_csv('ml-latest-small/movies.csv')	# original genre metadata

	#Construct a reverse map of indices and movie titles
	indices = pd.Series(ml_df.index, index=ml_df['title']).drop_duplicates()

	#clean genre data.
	features = ['genres']
	for feature in features:
		ml_df[feature] = ml_df[feature].apply(clean_genre_data)

	#clean cast data.
	features = ['casts']
	for feature in features:
		ml_df[feature] = ml_df[feature].apply(clean_cast_data)

	#split title data into year.
	features = ['title']
	for feature in features:
		ml_df['clean_title'] = ml_df[feature].apply(split_title)
		ml_df['year'] = ml_df[feature].apply(get_year)

	ml_df['soup'] = ml_df.apply(create_soup, axis=1)

	########################## original genre metadata ##########################

	print("\n\n Original MovieLens dataset using only genre : ")

	#split title data into year.
	features = ['title']
	for feature in features:
		og_df['clean_title'] = og_df[feature].apply(split_title)
		og_df['year'] = og_df[feature].apply(get_year)

	count = CountVectorizer(stop_words='english')
	count_matrix_og = count.fit_transform(og_df['genres'])
	cosine_sim_og = cosine_similarity(count_matrix_og, count_matrix_og)

	# Reset index of our main DataFrame and construct reverse mapping as before
	og_df = og_df.reset_index()
	indices = pd.Series(og_df.index, index=og_df['clean_title'])

	movie_rec = get_recommendations('Toy Story',cosine_sim_og)
	print(movie_rec)

	########################## genre/director/cast metadata ##########################

	print("\n\n Mined metadata using genre, director and cast  : ")

	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(ml_df['soup'])
	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	# Reset index of our main DataFrame and construct reverse mapping as before
	ml_df = ml_df.reset_index()
	indices = pd.Series(ml_df.index, index=ml_df['clean_title'])

	movie_rec = get_recommendations('Toy Story',cosine_sim)
	print(movie_rec)
