import pandas as pd 
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


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

def merge_soup_to_review(x):
	ret_val = str(x['soup']) + ' ' + str(x['review'])
	return ret_val

if __name__== "__main__":

	#loading datasets
	ml_df=pd.read_csv('movies-director-casts.csv')	# genre/director/cast metadata
	rv_df=pd.read_csv('movies-user-review.csv')	# review metadata

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

	########################## genre/director/cast metadata ##########################

	print("\n\nMined metadata using genre, director and cast  : ")

	#split title data into year.
	features = ['title']
	for feature in features:
		ml_df['clean_title'] = ml_df[feature].apply(split_title)
		ml_df['year'] = ml_df[feature].apply(get_year)

	ml_df['soup'] = ml_df.apply(create_soup, axis=1)

	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(ml_df['soup'])
	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	# Reset index of our main DataFrame and construct reverse mapping as before
	ml_df = ml_df.reset_index()
	indices = pd.Series(ml_df.index, index=ml_df['clean_title'])

	movie_rec = get_recommendations('Toy Story',cosine_sim)
	print("Input Movie : 'Toy Story'")
	print(movie_rec)

	########################## user review metadata using TFIDF ##########################

	print("\n\nMined metadata using user review using TFIDF : ")

	#split title data into year.
	features = ['title']
	for feature in features:
		rv_df['clean_title'] = rv_df[feature].apply(split_title)
		rv_df['year'] = rv_df[feature].apply(get_year)

	#Replace NaN with an empty string
	rv_df['review'] = rv_df['review'].fillna('')
	reviews = rv_df['review']

	#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
	tfidf = TfidfVectorizer(stop_words='english')

	#Construct the required TF-IDF matrix by fitting and transforming the data
	tfidf_matrix = tfidf.fit_transform(reviews)
	
	# Compute the cosine similarity matrix
	cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

	# Reset index of our main DataFrame and construct reverse mapping as before
	rv_df = rv_df.reset_index()
	indices = pd.Series(rv_df.index, index=rv_df['clean_title'])

	movie_rec = get_recommendations('Toy Story',cosine_sim)
	print("Input Movie : 'Toy Story'")
	print(movie_rec)

	########################## user review + genre/director/cast metadata using TFIDF ##########################

	print("\n\nMined metadata using user review + genre/director/cast : ")

	#merged_movielens_metadata = pd.merge(left=rv_df, right=ml_df, left_on='movieId', right_on='movieId')
	merged_movielens_metadata = pd.concat([ml_df, reviews], axis=1)

	merged_movielens_metadata['soup'] = merged_movielens_metadata.apply(merge_soup_to_review, axis=1)

	#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
	tfidf = TfidfVectorizer(stop_words='english')

	#Construct the required TF-IDF matrix by fitting and transforming the data
	tfidf_matrix = tfidf.fit_transform(merged_movielens_metadata['soup'])
	
	merge_feature_matrix = scipy.sparse.hstack([tfidf_matrix, count_matrix])

	# Compute the cosine similarity matrix
	cosine_sim = cosine_similarity(merge_feature_matrix, merge_feature_matrix)

	movie_rec = get_recommendations('Toy Story',cosine_sim)
	print("Input Movie : 'Toy Story'")
	print(movie_rec)
