#!/usr/bin/env python
# coding: utf-8

# # Neumf - Movielens data as implicit feedback

# # 1. Download data

# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '2.x')
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import pandas as pd
import numpy as np


# In[ ]:


with open("/content/drive/My Drive/Colab Notebooks/ratings.csv", 'rb') as rating:
  ratings = pd.read_csv(rating)

# Generate all_users and all_items, i.e. number of users and items 

ratings['user_id'] = ratings['userId'].astype("category").cat.codes
df_sort = ratings.sort_values(by=['userId', 'timestamp']).reset_index(drop=True)
all_users = list(np.sort(df_sort.userId.unique()))
all_items = list(np.sort(df_sort.movieId.unique()))


# # 2. Train test split

# In[ ]:


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
    df_test = df_test[['userId', 'user_id', 'movieId', 'binary_rating', 'rating']]

    # Remove the test set from the test set
    mask_test = df_test.index
    df_train = df_train.drop(mask_test)
    df_train = df_train[['userId', 'user_id', 'movieId', 'binary_rating', 'rating']]

    return df_train, df_test

k = 5 # instead of 5
df_train, df_test = train_test_split(df_sort,k)


# In[ ]:


# 110 items with cold start problem - 100K dataset
# 3 items with cold start problem - 1m dataset

df_train['item_id'] = df_train['movieId'].astype("category").cat.codes
item_list = df_train[['movieId', 'item_id']]
item_list = item_list.drop_duplicates()
df_test_new = pd.merge(df_test, item_list, how = 'left', left_on= 'movieId', right_on='movieId')


# # 3. Negative Sampling

# In[ ]:


# Create a dictionary contains watched movies for each user
training_dict = df_train.groupby('user_id')['item_id'].apply(lambda x: x.tolist())
training_dict = training_dict.to_dict()

# Function to random select unwatch movies
def movie_choice(user_id, n):
    choice = set(item_list['item_id']) - set(training_dict[user_id])
    rand_movies = np.random.choice(list(choice), n)
    return list(rand_movies)

def get_train_instances(train, training_dict, ratio):
    # Positive instances
    user_train = list(train['user_id'])
    item_train = list(train['item_id'])
    labels = [1] * len(user_train)

    # Negative instances
    for k, v in training_dict.items():
      num_negatives = len(v) * ratio # Define number of negative sampling by given ratio
      user_train.extend(([k] * num_negatives))
      item_train.extend(movie_choice(k, num_negatives))
      labels.extend(([0] * num_negatives))

    return user_train, item_train, labels


# # 4. Prediction function to obtain top_k_hits dataframe

# In[ ]:


def prediction(training_data, model = 'gmf_model', K = 5):
  df = training_data.iloc[:]
  df_items_unique = df['item_id'].unique()
  df_users_unique = df['user_id'].unique()
  print(len(df_users_unique))
  batch_size = 128
  top_k_df = pd.DataFrame()
  for users in range(len(df_users_unique)):
    users_df  = df_users_unique[users]
    movies_watched = df[df['user_id'] == users]['item_id'].unique()
    movies_notwatched = list(set(df_items_unique) - set(movies_watched)) # to be used for prediction

    test_user_input = np.repeat(users, len(movies_notwatched)).reshape(-1,1).astype('int64')
    test_item_input = np.array(movies_notwatched).reshape(-1,1).astype('int64')

  
    if model == 'gmf_model':
      pred_test = gmf_model.predict([test_user_input, test_item_input]) #, batch_size = batch_size
    elif model == 'mlp_model':
      pred_test = mlp_model.predict([test_user_input, test_item_input])
    elif model == 'neumf_model':
      pred_test = neumf_model.predict([test_user_input, test_item_input])
    d = {'pred_user_id': list(i[0] for i in test_user_input), 'Recommended_movieId': list(i[0] for i in test_item_input), 
        'prediction': list(i[0] for i in pred_test)}
    recommended_df = pd.DataFrame(data = d)
    top_k_items = recommended_df.sort_values(by='prediction', ascending = False)[:K]
    if users % 500 == 0 and users != 0:
      print("no. of users: ", users + 1)
    
    top_k_df = pd.concat([top_k_df, top_k_items])

  df_hit = pd.merge(df_test_new, top_k_df, how = 'left', left_on = ['user_id', 'item_id'], right_on = ['pred_user_id','Recommended_movieId'] )

  return top_k_df, df_hit, len(df_hit.dropna())


# 
# # 5(a) GMF model

# In[ ]:


# HYPERPARAMS

latent_features = 32

# Graph

tf.keras.backend.clear_session()

user_input = tf.keras.Input(shape=(1,), dtype='int64', name='user_gmf_input')
item_input = tf.keras.Input(shape = (1,), dtype = 'int64', name = 'item_gmf_input')
gmf_u_var = tf.keras.layers.Embedding(len(all_users), latent_features)(user_input)
gmf_i_var = tf.keras.layers.Embedding(len(item_list), latent_features)(item_input)
gmf_user_flatten = tf.keras.layers.Flatten()(gmf_u_var)
gmf_item_flatten = tf.keras.layers.Flatten()(gmf_i_var)
gmf_matrix = tf.keras.layers.multiply([gmf_user_flatten, gmf_item_flatten])

#dropout
#gmf_dropout = tf.keras.layers.Dropout(0.3)(gmf_matrix, training = True)

gmf_output = tf.keras.layers.Dense(1, activation = 'relu')(gmf_matrix)

gmf_model = tf.keras.Model([user_input, item_input], gmf_output)
gmf_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] ) #

gmf_model.summary()


# Training loop below, such that negative sampling for each epoch is random and different everytime

# In[ ]:


import evaluation_v2


epochs = 20
ratio = 3

for epoch in range(epochs):

  user_input, item_input, labels = get_train_instances(df_train, training_dict, ratio)
  
  gmf_model.fit([np.array(user_input), np.array(item_input)], np.array(labels),
                batch_size = 1024,  epochs=1, shuffle = True ) 
  
  print(epoch + 1)

top_k_df, df_hit, len_df_hit = prediction(df_train, 'gmf_model', K = 5)
print(len(df_hit.dropna().drop_duplicates()))

# Evaluation
import evaluation_v2
top_k_df['user_id'] = top_k_df['pred_user_id']
print("Precision at K", evaluation_v2.precision_at_k(df_test_new, top_k_df, k))
print("Recall at K", evaluation_v2.recall_at_k(df_test_new, top_k_df, k))
print("MAP", evaluation_v2.map_at_k(df_test_new, top_k_df, k))
print("NDCG", evaluation_v2.ndcg_at_k(df_test_new, top_k_df, k))


# In[ ]:


gmf_user_embeddings = gmf_model.layers[2].get_weights()[0]
gmf_item_embeddings = gmf_model.layers[3].get_weights()[0]
print(gmf_item_embeddings.shape,gmf_user_embeddings.shape)


# In[ ]:


import pickle
with open('gmf_item_embedding_neg_small.pickle', 'wb') as gmf_item:
    pickle.dump(gmf_item_embeddings, gmf_item)
with open('gmf_user_embeddings_neg_small.pickle', 'wb') as gmf_user:
    pickle.dump(gmf_user_embeddings, gmf_user)


# 
# # 5(b) MLP model

# In[ ]:


tf.keras.backend.clear_session()

latent_dimension = 32
#batch_size = 64

user_input = tf.keras.Input(shape=(1,), dtype='int32', name='user_mlp_input')
item_input = tf.keras.Input(shape = (1,), dtype = 'int32', name = 'item_mlp_input')
user_mlp_embed = tf.keras.layers.Embedding(len(all_users), latent_dimension,
                                           name = 'user_mlp_embed_layer', input_length=None)(user_input) #, input_length =1
item_mlp_embed = tf.keras.layers.Embedding(len(item_list), latent_dimension, 
                                           name = 'item_mlp_embed_layer', input_length=None)(item_input) #, input_length =1
user_mlp_flatten = tf.keras.layers.Flatten(name = 'user_embed_flatten')(user_mlp_embed)
item_mlp_flatten = tf.keras.layers.Flatten(name = 'item_embed_flatten')(item_mlp_embed)

mlp_join = tf.keras.layers.concatenate([user_mlp_flatten, item_mlp_flatten], axis = -1, name = 'mlp_concat_layer')
mlp_flatten = tf.keras.layers.Flatten(name = 'user_concat_layer_flatten')(mlp_join)
#mlp_dropout1 = tf.keras.layers.Dropout(0.3, name = 'mlp_first_dropout')(mlp_flatten, training = True)

mlp_dense1 = tf.keras.layers.Dense(16, activation='relu')(mlp_flatten)
#mlp_dropout2 = tf.keras.layers.Dropout(0.3)(mlp_dense1, training = True)

mlp_dense2 = tf.keras.layers.Dense(8, activation='relu')(mlp_dense1)
#mlp_dropout3 = tf.keras.layers.Dropout(0.3)(mlp_dense2, training = True)

mlp_dense3 = tf.keras.layers.Dense(4, activation='relu')(mlp_dense2)
#mlp_dropout4 = tf.keras.layers.Dropout(0.3)(mlp_dense3, training = True)

mlp_output = tf.keras.layers.Dense(1)(mlp_dense3) # output is a number

mlp_model = tf.keras.Model([user_input, item_input], mlp_output)
mlp_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #binary_crossentropy

mlp_model.summary()


# In[ ]:


epochs = 20
ratio = 1

for epoch in range(epochs):

  user_input, item_input, labels = get_train_instances(df_train, training_dict, ratio)
  
  mlp_model.fit([np.array(user_input), np.array(item_input)], np.array(labels),
                batch_size = 1024,  epochs=1, shuffle = True ) 
  
  print(epoch + 1)

top_k_df, df_hit, len_df_hit = prediction(df_train, 'mlp_model', K = 5)
print(len(df_hit.dropna().drop_duplicates()))

# Evaluation
import evaluation_v2
top_k_df['user_id'] = top_k_df['pred_user_id']
print("Precision at K", evaluation_v2.precision_at_k(df_test_new, top_k_df, k))
print("Recall at K", evaluation_v2.recall_at_k(df_test_new, top_k_df, k))
print("MAP", evaluation_v2.map_at_k(df_test_new, top_k_df, k))
print("NDCG", evaluation_v2.ndcg_at_k(df_test_new, top_k_df, k))


# In[ ]:


trained_gmf_users = gmf_model.layers[2].get_weights()[0]
trained_gmf_items = gmf_model.layers[3].get_weights()[0]

trained_mlp_users = mlp_model.layers[2].get_weights()[0]
trained_mlp_items = mlp_model.layers[3].get_weights()[0]

trained_mlp_dense0 = mlp_model.layers[8].get_weights()[0]
trained_mlp_dense1 = mlp_model.layers[9].get_weights()[0]
trained_mlp_dense2 = mlp_model.layers[10].get_weights()[0]


# In[ ]:


mlp_dense0 = mlp_model.layers[8].get_weights()[0]
mlp_dense1 = mlp_model.layers[9].get_weights()[0]
mlp_dense2 = mlp_model.layers[10].get_weights()[0]
import pickle
with open('mlp_dense0_neg_small.pickle', 'wb') as dense0:
    pickle.dump(mlp_dense0, dense0)
with open('mlp_dense1_neg_small.pickle', 'wb') as dense1:
    pickle.dump(mlp_dense1, dense1)
with open('mlp_dense2_neg_small.pickle', 'wb') as dense2:
    pickle.dump(mlp_dense2, dense2)
with open('mlp_user_embeddings_neg_small.pickle', 'wb') as mlp_user_embed_train:
    pickle.dump(mlp_user_embeddings, mlp_user_embed_train)
with open('mlp_item_embeddings_neg_small.pickle', 'wb') as mlp_item_embed_train:
    pickle.dump(mlp_item_embeddings, mlp_item_embed_train)


# 
# # 5(c) NeuMF model

# In[ ]:


with open("gmf_user_embeddings_neg.pickle", 'rb') as gmf_user:
  trained_gmf_users = pickle.load(gmf_user)
with open("gmf_item_embedding_neg.pickle", 'rb') as gmf_item:
  trained_gmf_items = pickle.load(gmf_item)
with open("mlp_user_embeddings.pickle", 'rb') as mlp_user:
  trained_mlp_users = pickle.load(mlp_user)
with open("mlp_item_embeddings.pickle", 'rb') as mlp_item:
  trained_mlp_items = pickle.load(mlp_item)
with open("mlp_dense0.pickle", 'rb') as dense0:
  trained_mlp_dense0 = pickle.load(dense0)
with open("mlp_dense1.pickle", 'rb') as dense1:
  trained_mlp_dense1 = pickle.load(dense1)
with open("mlp_dense2.pickle", 'rb') as dense2:
  trained_mlp_dense2 = pickle.load(dense2)


# In[ ]:


with open("/content/drive/My Drive/gmf_user_embeddings_neg_small.pickle", 'rb') as gmf_user:
  trained_gmf_users = pickle.load(gmf_user)
with open("/content/drive/My Drive/gmf_item_embedding_neg_small.pickle", 'rb') as gmf_item:
  trained_gmf_items = pickle.load(gmf_item)
with open("/content/drive/My Drive/mlp_user_embeddings_neg_small.pickle", 'rb') as mlp_user:
  trained_mlp_users = pickle.load(mlp_user)
with open("/content/drive/My Drive/mlp_item_embeddings_neg_small.pickle", 'rb') as mlp_item:
  trained_mlp_items = pickle.load(mlp_item)
with open("/content/drive/My Drive/mlp_dense0_neg_small.pickle", 'rb') as dense0:
  trained_mlp_dense0 = pickle.load(dense0)
with open("/content/drive/My Drive/mlp_dense1_neg_small.pickle", 'rb') as dense1:
  trained_mlp_dense1 = pickle.load(dense1)
with open("/content/drive/My Drive/mlp_dense2_neg_small.pickle", 'rb') as dense2:
  trained_mlp_dense2 = pickle.load(dense2)


# In[ ]:


print(len(item_list))
trained_mlp_items.shape


# In[ ]:


# https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/NeuMF.py

# need to pretain the gmf and mlp model, as per papaer
# then use the trained embeddings in the model below
# then use sgd, not Adam, as per the paper. 

# Things to do: to get the embedding layers for the 2 models, save them and add them in . 

tf.keras.backend.clear_session()

latent_dimension = 32
latent_features = 32
#batch_size = 64
tf.keras.backend.clear_session()

user_input = tf.keras.Input(shape=(1,), dtype='int64', name='user_input')
item_input = tf.keras.Input(shape = (1,), dtype = 'int64', name = 'item_input')

#gmf_u_var = tf.keras.layers.Embedding(len(all_users), latent_features, embeddings_initializer='uniform')
gmf_u_var = tf.keras.layers.Embedding(len(all_users), latent_features, embeddings_initializer = tf.keras.initializers.Constant(trained_gmf_users), name = 'user_gmf_embed_layer', trainable = False)(user_input)
gmf_i_var = tf.keras.layers.Embedding(len(item_list), latent_features, embeddings_initializer = tf.keras.initializers.Constant(trained_gmf_items), name = 'item_gmf_embed_layer', trainable = False)(item_input)
user_mlp_embed = tf.keras.layers.Embedding(len(all_users), latent_dimension, embeddings_initializer= tf.keras.initializers.Constant(trained_mlp_users)
                                           , trainable = False)(user_input) #, input_length =1
item_mlp_embed = tf.keras.layers.Embedding(len(item_list), latent_dimension, embeddings_initializer= tf.keras.initializers.Constant(trained_mlp_items),
                                           name = 'trained_mlp_items', trainable = False)(item_input) #, input_length =1

gmf_user_flatten = tf.keras.layers.Flatten()(gmf_u_var)
gmf_item_flatten = tf.keras.layers.Flatten()(gmf_i_var)
gmf_matrix = tf.keras.layers.multiply([gmf_user_flatten, gmf_item_flatten])

user_mlp_flatten = tf.keras.layers.Flatten(name = 'user_embed_flatten')(user_mlp_embed)
item_mlp_flatten = tf.keras.layers.Flatten(name = 'item_embed_flatten')(item_mlp_embed)
mlp_join = tf.keras.layers.concatenate([user_mlp_flatten, item_mlp_flatten], axis = -1, name = 'mlp_concat_layer')
mlp_flatten = tf.keras.layers.Flatten(name = 'user_concat_layer_flatten')(mlp_join)

#mlp_dropout1 = tf.keras.layers.Dropout(0.3, name = 'mlp_first_dropout')(mlp_flatten, training = True)
mlp_dense1 = tf.keras.layers.Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.Constant(trained_mlp_dense0), trainable = False)(mlp_flatten) #
#mlp_dropout2 = tf.keras.layers.Dropout(0.3)(mlp_dense1, training = True)
mlp_dense2 = tf.keras.layers.Dense(8, activation='relu', kernel_initializer=tf.keras.initializers.Constant(trained_mlp_dense1), trainable = False)(mlp_dense1) #, trainable = False
#mlp_dropout3 = tf.keras.layers.Dropout(0.3)(mlp_dense2, training = True)
mlp_dense3 = tf.keras.layers.Dense(4, activation='relu', kernel_initializer=tf.keras.initializers.Constant(trained_mlp_dense2), trainable = False)(mlp_dense2) #, trainable = False
#mlp_dropout4 = tf.keras.layers.Dropout(0.3)(mlp_dense3, training = True)
#mlp_output = tf.keras.layers.Dense(1)(mlp_dense3) # output is a number

neumf_input = tf.keras.layers.concatenate([gmf_matrix, mlp_dense3])
neumf_output = tf.keras.layers.Dense(1)(neumf_input)


### From here, use an approach similar to the paper, where even the final layer of gmf and mlp are used. 
###---------------------------------------------------------------------------------######

#gmf_prediction = tf.keras.layers.Dense(1)(gmf_matrix)

#mlp_prediction = tf.keras.layers.Dense(1)(mlp_dense3)

#neumf_input = tf.keras.layers.concatenate([0.5* gmf_prediction, 0.5 * mlp_prediction]) 
#neumf_flatten = tf.keras.layers.Flatten(name = 'neumf_concat_layer_flatten')(neumf_input)
#neumf_output = tf.keras.layers.Dense(1)(neumf_flatten)
###---------------------------------------------------------------------------------######


neumf_model = tf.keras.Model([user_input, item_input], neumf_output)
neumf_model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

neumf_model.summary()


# In[ ]:


epochs = 20
ratio = 7

for epoch in range(epochs):

  user_input, item_input, labels = get_train_instances(df_train, training_dict, ratio)
  
  neumf_model.fit([np.array(user_input), np.array(item_input)], np.array(labels),
                batch_size = 1024,  epochs=1, shuffle = True ) 
  
  print(epoch + 1)

top_k_df, df_hit, len_df_hit = prediction(df_train, 'neumf_model', K = 5)
print(len(df_hit.dropna().drop_duplicates()))

# Evaluation
import evaluation_v2
top_k_df['user_id'] = top_k_df['pred_user_id']
print("Precision at K", evaluation_v2.precision_at_k(df_test_new, top_k_df, k))
print("Recall at K", evaluation_v2.recall_at_k(df_test_new, top_k_df, k))
print("MAP", evaluation_v2.map_at_k(df_test_new, top_k_df, k))
print("NDCG", evaluation_v2.ndcg_at_k(df_test_new, top_k_df, k))

