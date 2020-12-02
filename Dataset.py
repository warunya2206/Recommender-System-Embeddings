#!/usr/bin/env python
# coding: utf-8




import pickle
import numpy as np

with open("/content/drive/My Drive/1m_ratings.csv", 'rb') as rating_1m:
  ratings = pickle.load(rating_1m)




# Generate all_users and all_items, i.e. number of users and items 
ratings['user_id'] = ratings['userId'].astype("category").cat.codes
df_sort = ratings.sort_values(by=['userId', 'timestamp']).reset_index(drop=True)
all_users = list(np.sort(df_sort.userId.unique()))
all_items = list(np.sort(df_sort.movieId.unique()))





def train_test_split(df,n):
    """
    Splits our original data into one test and one
    training set. 
    The test set is made up of one item for each user. This is
    our holdout item used to compute Top@K later.
    The training set is the same as our original data but
    without any of the holdout items.
    Args:
        df (dataframe): Our original data
    Returns:
        df_train (dataframe): All of our data except holdout items
        df_test (dataframe): Only our holdout items.
    """

    # Create two copies of our dataframe that we can modify
    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    # Group by user_id and select only the last n item
    # Test dataframe
    df_test = df_test.groupby(['user_id']).tail(n)
    df_test = df_test[['userId', 'user_id', 'movieId', 'rating']]

    # Remove the test set from the test set
    mask_test = df_test.index
    df_train = df_train.drop(mask_test)
    df_train = df_train[['userId', 'user_id', 'movieId', 'rating']]

    return df_train, df_test

k = 5
df_train, df_test = train_test_split(df_sort,k)





# 110 items with cold start problem

df_train['item_id'] = df_train['movieId'].astype("category").cat.codes
df_train

item_list = df_train[['movieId', 'item_id']]
item_list = item_list.drop_duplicates()





# Read movies dataset
rnames = ['movie_id','title','genres']

movies = pd.read_table('/content/drive/My Drive/movies.dat',sep='::',header=None, names=rnames)
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year # As there are some NaN years, resulting type will be float (decimals)
movies['genre'] = movies.genres.str.split('|')





# Find out how many categories is each movie grouped under
genre_count = [len(i) for i in movies.genre]
genre_count_df = pd.DataFrame(genre_count, columns = ['genre_count'])
movies1 = pd.concat([movies, genre_count_df], axis = 1)
movies1['categories_movie_isin'] = movies1['genre_count'].astype(str) + "_genre_category"
movies_info = pd.merge(movies, movies1, left_on = 'movie_id', right_on = 'movie_id', how = 'left')
movies_info = pd.merge(item_list, movies1, left_on = 'movieId', right_on = 'movie_id', how = 'left')
movies_info.year = movies_info.year.astype(str)
movies_info = movies_info.drop(columns=['movie_id','genre'])





# Read trained embeddings
with open("/content/drive/My Drive/gmf_item_embedding_neg.pickle", 'rb') as gmf_item:
  trained_gmf_items = pickle.load(gmf_item)
with open("/content/drive/My Drive/mlp_item_embeddings_neg.pickle", 'rb') as mlp_item:
  trained_mlp_items = pickle.load(mlp_item)




# Create combine data set
# Using mlp items - implicit feedback
dataset = pd.DataFrame(trained_mlp_items)
dataset['item_id'] = dataset.index
dataset = pd.merge(dataset, movies_info, left_on = 'item_id', right_on = 'item_id').dropna()
dataset['Label'] = pd.factorize(dataset['genres'])[0] # Create LabelEncoder
label_code_dict = dict(zip(dataset['Label'], dataset['genres'])) # Create dict to map LabelEncoder





# Create combine data set
# Using gmf items - implicit feedback
dataset = pd.DataFrame(trained_gmf_items)
dataset['item_id'] = dataset.index
dataset = pd.merge(dataset, movies_info, left_on = 'item_id', right_on = 'item_id').dropna()
dataset['Label'] = pd.factorize(dataset['genres'])[0] # Create LabelEncoder
label_code_dict = dict(zip(dataset['Label'], dataset['genres'])) # Create dict to map LabelEncoder

