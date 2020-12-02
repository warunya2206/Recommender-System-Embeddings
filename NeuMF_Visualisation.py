#!/usr/bin/env python
# coding: utf-8

# ## **Retrieve category codings for users and items**

# In[ ]:


import pickle
with open("/content/drive/My Drive/1m_ratings.csv", 'rb') as rating_1m:
  ratings = pickle.load(rating_1m)


# In[ ]:


# Generate all_users and all_items, i.e. number of users and items 
import numpy as np
ratings['user_id'] = ratings['userId'].astype("category").cat.codes
df_sort = ratings.sort_values(by=['userId', 'timestamp']).reset_index(drop=True)
all_users = list(np.sort(df_sort.userId.unique()))
all_items = list(np.sort(df_sort.movieId.unique()))
print("Number of users: ", len(all_users))
print("Number of movies: ", len(all_items))


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
    df_test = df_test[['userId', 'user_id', 'movieId', 'rating']]

    # Remove the test set from the test set
    mask_test = df_test.index
    df_train = df_train.drop(mask_test)
    df_train = df_train[['userId', 'user_id', 'movieId', 'rating']]

    return df_train, df_test

k = 5
df_train, df_test = train_test_split(df_sort,k)


# Only items in the df_train are category coded, not the entire dataset, as the model is only trained on df_train dataset

# In[ ]:


# 110 items with cold start problem

df_train['item_id'] = df_train['movieId'].astype("category").cat.codes
df_train

item_list = df_train[['movieId', 'item_id']]
print(len(item_list))
item_list = item_list.drop_duplicates()
print(len(item_list))
item_list.head(2)


# ## **Download movie and users additional information**

# In[ ]:


#movies = pd.read_csv("/content/drive/My Drive/movies.dat")
import pandas as pd
rnames = ['movie_id','title','genres']

movies = pd.read_table('/content/drive/My Drive/movies.dat',sep='::',header=None, names=rnames)
movies.head(2)


# In[ ]:


#movies = pd.read_csv("/content/drive/My Drive/movies.dat")
#UserID::Gender::Age::Occupation::Zip-code
rnames = ['UserID','Gender','Age', 'Occupation', 'Zipcode']

users_demo = pd.read_table('/content/drive/My Drive/users.dat',sep='::',header=None, names=rnames)
users_demo.head(2)


# ## **Grouping of user's demographic information**

# In[ ]:


users_occupation_ind2wd = {0:  "other", 1:  "academic/educator", 2:  "artist", 3:  "clerical/admin", 
                    4:  "college/grad student", 5:  "customer service", 6:  "doctor/health care", 
                    7:  "executive/managerial", 8:  "farmer", 9:  "homemaker" , 10:  "K-12 student", 
                    11:  "lawyer", 12:  "programmer", 13:  "retired", 14:  "sales/marketing", 
                    15:  "scientist", 16:  "self-employed", 17:  "technician/engineer", 18:  "tradesman/craftsman", 
                    19:  "unemployed", 20:  "writer"}

occupation_df = pd.DataFrame(list([x,y]) for x,y in users_occupation_ind2wd.items())
occupation_df = occupation_df.rename(columns = {0:'occupation_ind', 1: 'occupation'})
occupation_df.head()


# In[ ]:


# create cat codes column
users_demo['user_id'] = users_demo['UserID'].astype("category").cat.codes

# Add column Occupation
users_info = pd.merge(users_demo, occupation_df, left_on = 'Occupation', right_on = 'occupation_ind', how = 'left' )

# Add column group age
users_info['age_group'] = np.where(users_info.Age<18, 'Under 18', 
                     np.where(users_info.Age<=24, '18-24', 
                     np.where(users_info.Age<=34, '25-34',
                     np.where(users_info.Age<=44, '35-44',
                     np.where(users_info.Age<=49, '45-49',
                     np.where(users_info.Age<=55, '50-55',
                     np.where(users_info.Age>55, '56+','seniors')))))))

users_info = users_info.drop(columns=['Occupation'])

# Add combine column
users_info['age_gender'] = users_info.Gender + " - " + users_info.age_group
users_info.head()


# In[ ]:


user_rating = ratings.pivot_table(index = 'userId',  values = 'rating', aggfunc=np.count_nonzero).reset_index().rename(columns = {'rating':'count_ratings'})
user_avg_rating = ratings.pivot_table(index = 'userId',  values = 'rating', aggfunc=np.average).reset_index().rename(columns = {'rating':'avg_ratings'})
user_ratings = pd.merge(user_rating, user_avg_rating, left_on = 'userId', right_on = 'userId')
users_info_top = pd.merge(users_info, user_ratings, left_on = 'UserID', right_on = 'userId')
users_info_top = users_info_top.sort_values(by=['occupation', 'count_ratings'], ascending = False) 
users_info_top = users_info_top.groupby("occupation").head(50)
users_info_top


# ## **Grouping of movie's information**

# In[ ]:


movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year # As there are some NaN years, resulting type will be float (decimals)
movies['genre'] = movies.genres.str.split('|')
movies.head()


# In[ ]:


# Find out how many categories is each movie grouped under
genre_count = [len(i) for i in movies.genre]
genre_count_df = pd.DataFrame(genre_count, columns = ['genre_count'])
movies1 = pd.concat([movies, genre_count_df], axis = 1)
movies1['categories_movie_isin'] = movies1['genre_count'].astype(str) + "_genre_category"
movies_info = pd.merge(movies, movies1, left_on = 'movie_id', right_on = 'movie_id', how = 'left')


# In[ ]:


movies_info = pd.merge(item_list, movies1, left_on = 'movieId', right_on = 'movie_id', how = 'left')
movies_info['year_group'] = np.where(movies_info.year<=1960, '< 1960 ',
                                    np.where(movies_info.year<=1973, '< 1973',
                                    np.where(movies_info.year<=1982, '< 1982',
                                    np.where(movies_info.year<=1989, '< 1989',
                                    np.where(movies_info.year<=1992, '< 1992',         
                                    np.where(movies_info.year<=1993, '< 1993',
                                    np.where(movies_info.year<=1994, '< 1994',
                                    np.where(movies_info.year<=1995, '< 1995',
                                    np.where(movies_info.year<=1996, '< 1996',
                                    np.where(movies_info.year<=1997, '< 1997',
                                    np.where(movies_info.year<=1998, '< 1998',
                                    np.where(movies_info.year<=1999, '< 1999',
                                    np.where(movies_info.year<=2000, '< 2000','No details')))))))))))))
movies_info['year_genre'] = movies_info['year_group'] + " - " + movies_info['genres']
movies_info.year = movies_info.year.astype(str)
movies_info = movies_info.drop(columns=['movie_id','genre'])
movies_info.head(3)


# In[ ]:


# Merge with information from rating dataset
movies_rating = ratings.pivot_table(index = 'movieId',  values = 'rating', aggfunc=np.count_nonzero).reset_index().rename(columns = {'rating':'count_ratings'}) # Count ratings
avg_rating = ratings.pivot_table(index = 'movieId',  values = 'rating', aggfunc=np.average).reset_index().rename(columns = {'rating':'avg_ratings'}) # Average rating
movies_ratings = pd.merge(movies_info, movies_rating, left_on = 'movieId', right_on = 'movieId', how = 'left') # Merge with count_ratings infomation
movies_ratings = pd.merge(movies_ratings, avg_rating, left_on = 'movieId', right_on = 'movieId', how = 'left') # Merge with avg_ratings infomation
movies_ratings['rating_group'] = np.where(movies_ratings.avg_ratings < 3.75, 'Low',  
                                    np.where(movies_ratings.avg_ratings >= 3.75, 'high','No details'))
movies_ratings['rating_genre'] = movies_ratings['rating_group'] + " - " + movies_ratings['genres']

# Get top 100 movies for each genres
movies_ratings_top = movies_ratings.loc[movies_ratings['genre_count'].isin([1])] # Get only movies that have 1 genre
movies_ratings_top = movies_ratings_top.sort_values(by=['genres', 'count_ratings'], ascending=False) # Sort by count_ratings
movies_ratings_top = movies_ratings_top.groupby("genres").head(100) # Get top 100 from each genres
movies_ratings_top.head()


# In[ ]:


# Filter some gernes
movies_ratings = movies_ratings.sort_values(by=['rating_group', 'count_ratings'], ascending=False) # Sort by rating gruop and counts
movie_avg_rating_diff = movies_ratings.loc[movies_ratings['genres'].isin(['Documentary','Horror','Sci-Fi', 'War' ,"Children's|Comedy"])] # Select some unique genres
movie_avg_rating_top3 = movies_ratings.loc[movies_ratings['genres'].isin(['Drama','Horror','Comedy'])] # Select Top 3 genres
movie_avg_rating_top3 = movie_avg_rating_top3.groupby("genres").head(200)
movie_avg_rating_top3.head()


# ## **Read trained embeddings**

# In[ ]:


import pickle
with open("/content/drive/My Drive/gmf_user_embeddings.pickle", 'rb') as gmf_user:
  trained_gmf_users = pickle.load(gmf_user)
with open("/content/drive/My Drive/gmf_item_embedding.pickle", 'rb') as gmf_item:
  trained_gmf_items = pickle.load(gmf_item)
with open("/content/drive/My Drive/mlp_user_embeddings.pickle", 'rb') as mlp_user:
  trained_mlp_users = pickle.load(mlp_user)
with open("/content/drive/My Drive/mlp_item_embeddings.pickle", 'rb') as mlp_item:
  trained_mlp_items = pickle.load(mlp_item)
#with open("/content/drive/My Drive/mlp_dense0.pickle", 'rb') as dense0:
#  trained_mlp_dense0 = pickle.load(dense0)
#with open("/content/drive/My Drive/mlp_dense1.pickle", 'rb') as dense1:
#  trained_mlp_dense1 = pickle.load(dense1)
#with open("/content/drive/My Drive/mlp_dense2.pickle", 'rb') as dense2:
#  trained_mlp_dense2 = pickle.load(dense2)


# In[ ]:


import pickle
with open("/content/drive/My Drive/gmf_user_embeddings_neg.pickle", 'rb') as gmf_user:
  trained_gmf_users = pickle.load(gmf_user)
with open("/content/drive/My Drive/gmf_item_embedding_neg.pickle", 'rb') as gmf_item:
  trained_gmf_items = pickle.load(gmf_item)
with open("/content/drive/My Drive/mlp_user_embeddings_neg.pickle", 'rb') as mlp_user:
  trained_mlp_users = pickle.load(mlp_user)
with open("/content/drive/My Drive/mlp_item_embeddings_neg.pickle", 'rb') as mlp_item:
  trained_mlp_items = pickle.load(mlp_item)
#with open("/content/drive/My Drive/mlp_dense0.pickle", 'rb') as dense0:
#  trained_mlp_dense0 = pickle.load(dense0)
#with open("/content/drive/My Drive/mlp_dense1.pickle", 'rb') as dense1:
#  trained_mlp_dense1 = pickle.load(dense1)
#with open("/content/drive/My Drive/mlp_dense2.pickle", 'rb') as dense2:
#  trained_mlp_dense2 = pickle.load(dense2)


# In[ ]:


trained_mlp_items


# # **Run T-SNE**

# In[ ]:


from sklearn.manifold import TSNE
#bokeh for interactive plot and to save the result
from bokeh.plotting import figure, show, output_notebook, save, output_file
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource
from bokeh import palettes
from bokeh.palettes import Spectral, Spectral4, Spectral6, Category20, Turbo256, Dark2, inferno, Plasma256, Plasma, Paired

from bokeh.transform import factor_cmap
from bokeh.models.widgets import Tabs, Panel

output_notebook()


# In[ ]:


# choose any of the 4 embedding layers below:
# 'trained_gmf_users', 'trained_gmf_items', 'trained_mlp_users', 'trained_mlp_items'

def visualise(embedding_layer, n_components=2, perplexity=30, n_iter=1000, learning_rate=10):
  tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000,learning_rate=10)
  tsne_results = tsne.fit_transform(embedding_layer)
  
  return tsne_results

gmf_user_2d = visualise(trained_gmf_users)
gmf_item_2d = visualise(trained_gmf_items)
mlp_user_2d = visualise(trained_mlp_users)
mlp_item_2d = visualise(trained_mlp_items)


# # **Users' Visualisation**

# In[ ]:


def tsne_user_visual(tsne_results, name, dataset):
  '''input: Embedding vector to visualize and the name of the embedding vector.
  output: interactive plot with user id labeled on each point. The plot is saved in html file and can be download from colab'''

  df_combine = pd.DataFrame([i for i in range(len(tsne_results[:,0]))])
  df_combine.columns = ['user_id']
  df_combine['x-tsne'] = tsne_results[:,0]
  df_combine['y-tsne'] = tsne_results[:,1] 
  df_combine = pd.merge(df_combine, dataset, left_on = 'user_id', right_on = 'user_id').dropna()

  source = ColumnDataSource(dict(
      x = df_combine['x-tsne'],
      y = df_combine['y-tsne'],
      occupation = df_combine['occupation'],
      age_group = df_combine['age_group'],
      gender = df_combine['Gender'],
      age_gender = df_combine['age_gender'],
 ))
  
  # Define palette
  unique_occupation = df_combine['occupation'].unique()
  unique_age_group = df_combine['age_group'].unique()
  unique_Gender = df_combine['Gender'].unique()
  unique_age_gender = df_combine['age_gender'].unique()

  def define_palette(len_cluster):
    if len_cluster <= 2:
      palette = Dark2[3]
    elif len_cluster <= 8:
      palette = Dark2[len_cluster]
    elif len_cluster <= 20:
      palette = Category20[len_cluster]
    else:
      try:
        palette = inferno(len_cluster)
      except ValueError:
        palette = inferno(256)
    return palette
  
  title = 'T-SNE visualization of embeddings '+ name

  plot_gender = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_gender.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('gender', palette = define_palette(len(unique_Gender)), factors = unique_Gender),
                legend_group='gender')
  
  plot_agegrp = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_agegrp.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('age_group', palette = define_palette(len(unique_age_group)), factors = unique_age_group),
                legend_group='age_group')

  plot_occupation = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_occupation.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('occupation', palette = define_palette(len(unique_occupation)), factors = unique_occupation),
                legend_group='occupation')
  
  plot_agegender = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_agegender.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('age_gender', palette = define_palette(len(unique_age_gender)), factors = unique_age_gender),
                legend_group='age_gender')


  # hover tools
  gender_hover = plot_gender.select(dict(type=HoverTool))
  gender_hover.tooltips = {"content": "Gender: @gender"}
  plot_gender.legend.location = "top_left"
  plot_gender.legend.orientation = "horizontal"
  plot_gender.legend.click_policy="hide"

  age_hover = plot_agegrp.select(dict(type=HoverTool))
  age_hover.tooltips = {"content": "Age: @age_group"}
  plot_agegrp.legend.location = "top_left"
  plot_agegrp.legend.orientation = "horizontal"
  plot_agegrp.legend.click_policy="hide"

  occupation_hover = plot_occupation.select(dict(type=HoverTool))
  occupation_hover.tooltips = {"content": "Occupation: @occupation"}
  plot_occupation.legend.location = "top_left"
  plot_occupation.legend.orientation = "horizontal"
  plot_occupation.legend.click_policy="hide"

  agegender_hover = plot_agegender.select(dict(type=HoverTool))
  agegender_hover.tooltips = {"content": "Age_Gender: @age_gender"}
  plot_agegender.legend.location = "top_left"
  plot_agegender.legend.orientation = "horizontal"
  plot_agegender.legend.click_policy="hide"


# Create two panels, one for each conference
  By_Gender_panel = Panel(child=plot_gender, title='By Gender')
  By_Age_panel = Panel(child=plot_agegrp, title='By Age')
  By_Occupation_panel = Panel(child=plot_occupation, title='By Occupation')
  By_AgeGender_panel = Panel(child=plot_agegender, title='By Age & Gender')

  # Assign the panels to Tabs
  tabs = Tabs(tabs=[By_Gender_panel, By_Age_panel, By_Occupation_panel, By_AgeGender_panel])

  # Show the tabbed layout
  show(tabs)


# In[ ]:


tsne_user_visual(gmf_user_2d, 'gmf_user_embeddings', users_info)


# In[ ]:


tsne_user_visual(gmf_user_2d, 'gmf_user_embeddings', users_info_top)


# In[ ]:


tsne_user_visual(mlp_user_2d, 'mlp_user_embeddings', users_info)


# In[ ]:


tsne_user_visual(mlp_user_2d, 'mlp_user_embeddings', users_info_top)


# # **Movies Visualisation**

# In[ ]:


def tsne_item_visual(tsne_results, name, movie_dataset):
  '''input: Embedding vector to visualize and the name of the embedding vector.
  output: interactive plot with user id labeled on each point. The plot is saved in html file and can be download from colab'''

  df_combine = pd.DataFrame([i for i in range(len(tsne_results[:,0]))])
  df_combine.columns = ['item_id']
  df_combine['x-tsne'] = tsne_results[:,0]
  df_combine['y-tsne'] = tsne_results[:,1] 
  df_combine = pd.merge(df_combine, movie_dataset, left_on = 'item_id', right_on = 'item_id').dropna()

  source = ColumnDataSource(dict(
      item_id = df_combine['item_id'],
      x = df_combine['x-tsne'],
      y = df_combine['y-tsne'],
      genres = df_combine['genres'],
      year = df_combine['year'],
      year_group = df_combine['year_group'],
      title = df_combine['title'],
      year_genre = df_combine['year_genre'],
      rating_group = df_combine['rating_group'],
      rating_genre = df_combine['rating_genre']
 ))
  
  # Define palette
  unique_genre = df_combine['genres'].unique()
  unique_year = df_combine['year'].unique()
  unique_year_group = df_combine['year_group'].unique()
  unique_year_genre = df_combine['year_genre'].unique()
  unique_rating_group = df_combine['rating_group'].unique()
  unique_rating_genre = df_combine['rating_genre'].unique()

  def define_palette(len_cluster):
    if len_cluster <= 2:
      palette = Dark2[3]
    elif len_cluster <= 8:
      palette = Dark2[len_cluster]
    elif len_cluster <= 20:
      palette = Category20[len_cluster]
    else:
      try:
        palette = inferno(len_cluster)
      except ValueError:
        palette = inferno(256)
    return palette

  title = 'T-SNE visualization of embeddings '+ name

  plot_rating_group = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_rating_group.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color = factor_cmap('rating_group', palette = define_palette(len(unique_rating_group)), factors = unique_rating_group),
                legend_group='rating_group')

  plot_genre = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_genre.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('genres', palette = define_palette(len(unique_genre)), factors = unique_genre),
                legend_group='genres')
  
  plot_rating_genre = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_rating_genre.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color = factor_cmap('rating_genre', palette = define_palette(len(unique_rating_genre)), factors = unique_rating_genre),
                legend_group='rating_genre')
  
  plot_year = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_year.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('year', palette = define_palette(len(unique_year)), factors = unique_year),
                legend_group='year')

  plot_year_group = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_year_group.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('year_group', palette = define_palette(len(unique_year_group)), factors = unique_year_group),
                legend_group='year_group')

  plot_year_genre = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_year_genre.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('year_genre', palette = define_palette(len(unique_year_genre)), factors = unique_year_genre),
                legend_group='year_genre')
  
  # hover tools
  rating_group_hover = plot_rating_group.select(dict(type=HoverTool))
  rating_group_hover.tooltips = {"content": "Movie: @title"}
  plot_rating_group.legend.location = "top_left"
  plot_rating_group.legend.orientation = "horizontal"
  plot_rating_group.legend.click_policy="hide"

  rating_genre_hover = plot_rating_genre.select(dict(type=HoverTool))
  rating_genre_hover.tooltips = {"content": "Movie: @title"}
  plot_rating_genre.legend.location = "top_left"
  plot_rating_genre.legend.orientation = "horizontal"
  plot_rating_genre.legend.click_policy="hide"

  genre_hover = plot_genre.select(dict(type=HoverTool))
  genre_hover.tooltips = {"content": "Movie: @title itemid: @item_id"}
  plot_genre.legend.location = "top_left"
  plot_genre.legend.orientation = "horizontal"
  plot_genre.legend.click_policy="hide"

  year_hover = plot_year.select(dict(type=HoverTool))
  year_hover.tooltips = {"content": "Movie: @title"}
  plot_year.legend.location = "top_left"
  plot_year.legend.orientation = "horizontal"
  plot_year.legend.click_policy="hide"

  year_grp_hover = plot_year_group.select(dict(type=HoverTool))
  year_grp_hover.tooltips = {"content": "Movie: @title"}
  plot_year_group.legend.location = "top_left"
  plot_year_group.legend.orientation = "horizontal"
  plot_year_group.legend.click_policy="hide"

  year_genre_hover = plot_year_genre.select(dict(type=HoverTool))
  year_genre_hover.tooltips = {"content": "Movie: @title"}
  plot_year_genre.legend.location = "top_left"
  plot_year_genre.legend.orientation = "horizontal"
  plot_year_genre.legend.click_policy="hide"

# Create two panels, one for each conference
  By_rating_group = Panel(child=plot_rating_group, title='By grouped avg rating')
  By_rating_genre = Panel(child=plot_rating_genre, title='By Rating & Genre')
  By_genre = Panel(child=plot_genre, title='By Genre')
  By_year = Panel(child=plot_year, title='By Year')
  By_grp_year = Panel(child=plot_year_group, title='By Grp Year')
  By_year_genre = Panel(child=plot_year_genre, title='By Year & Genre')

  # Assign the panels to Tabs
  tabs = Tabs(tabs=[By_rating_group, By_genre, By_rating_genre, By_year, By_grp_year, By_year_genre])

  # Show the tabbed layout
  show(tabs)


# In[ ]:


# All movies
tsne_item_visual(gmf_item_2d, 'gmf_item_embeddings', movies_ratings)


# In[ ]:


# All movies
tsne_item_visual(mlp_item_2d, 'mlp_item_embeddings', movies_ratings)


# In[ ]:


# Get Top 100 movies (based on number of rated) from each genres
tsne_item_visual(gmf_item_2d, 'gmf_item_embeddings', movies_ratings_top)


# In[ ]:


# Get Top 100 movies (based on number of rated) from each genres
tsne_item_visual(mlp_item_2d, 'mlp_item_embeddings', movies_ratings_top)


# In[ ]:


# Cluster by average rating and selected only top 3 genre
tsne_item_visual(gmf_item_2d, 'gmf_item_embeddings', movie_avg_rating_top3)


# In[ ]:


# Cluster by average rating and selected only top 3 genre
tsne_item_visual(mlp_item_2d, 'mlp_item_embeddings', movie_avg_rating_top3)


# In[ ]:


#5 distinct movies genres
tsne_item_visual(gmf_item_2d, 'gmf_item_embeddings', movie_avg_rating_diff)


# In[ ]:


#5 distinct movies genres
tsne_item_visual(mlp_item_2d, 'mlp_item_latent', movie_avg_rating_diff)


# In[ ]:


##specify which genres to visualize
drama_df = movies_info.loc[movies_info['genres']=='Drama']
comedy_df = movies_info.loc[movies_info['genres']=='Comedy']
horror_df = movies_info.loc[movies_info['genres']=='Horror']
genre_df = drama_df.append(comedy_df).append(horror_df)
genre_df


# In[ ]:



def tsne_item_visual(tsne_results, name, no_of_genres = 1):
  '''input: Embedding vector to visualize and the name of the embedding vector.
  output: interactive plot with user id labeled on each point. The plot is saved in html file and can be download from colab'''

  df_combine = pd.DataFrame([i for i in range(len(tsne_results[:,0]))])
  df_combine.columns = ['item_id']
  df_combine['x-tsne'] = tsne_results[:,0]
  df_combine['y-tsne'] = tsne_results[:,1] 
  df_combine = pd.merge(df_combine, genre_df, how = 'left', left_on = 'item_id', right_on = 'item_id').dropna() 
  #df_combine = df_combine[df_combine.genre_count == no_of_genres]
  
  source = ColumnDataSource(dict(
      x = df_combine['x-tsne'],
      y = df_combine['y-tsne'],
      genres = df_combine['genres'],
      year= df_combine['year'],
      year_group = df_combine['year_group'],
      title = df_combine['title'],
      year_genre = df_combine['year_genre'],
      genre_count = df_combine['genre_count']
 ))
  
  title = 'T-SNE visualization of embeddings '+ name

  combination = len(df_combine['genres'].unique())
  plot_genre = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_genre.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('genres', palette=inferno(combination), factors = df_combine['genres'].unique()),
                legend_group = 'genres')
  
  combination = len(df_combine['year'].unique())
  plot_year = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_year.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('year', palette=inferno(combination), factors = df_combine['year'].unique()),
                legend_group = 'year')
  
  combination = len(df_combine['year_group'].unique())
  plot_year_group = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_year_group.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('year_group', palette=inferno(combination), factors = df_combine['year_group'].unique()),
                legend_group='year_group')

  combination = len(df_combine['year_genre'].unique())
  plot_year_genre = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover",
                     x_axis_type=None, y_axis_type=None, min_border=1)
  plot_year_genre.scatter(x='x', y='y', source=source,
                alpha=0.7, size=10, fill_color=factor_cmap('year_genre', palette=inferno(combination), factors = df_combine['year_genre'].unique()),
                legend_group='year_genre')
  
 
#fill_color=factor_cmap('gender', palette=Spectral6, factors = df_combine['Gender']
  # hover tools
  genre_hover = plot_genre.select(dict(type=HoverTool))
  genre_hover.tooltips = {"content": "Movie: @title Genre: @genres"}
  plot_genre.legend.location = "top_left"
  plot_genre.legend.orientation = "horizontal"
  plot_genre.legend.click_policy="hide"

  year_hover = plot_year.select(dict(type=HoverTool))
  year_hover.tooltips = {"content": "Movie: @title Year: @year"}
  plot_year.legend.location = "top_left"
  plot_year.legend.orientation = "horizontal"
  plot_year.legend.click_policy="hide"

  year_grp_hover = plot_year_group.select(dict(type=HoverTool))
  year_grp_hover.tooltips = {"content": "Movie: @title Grp_Year: @year_group"}
  plot_year_group.legend.location = "top_left"
  plot_year_group.legend.orientation = "horizontal"
  plot_year_group.legend.click_policy="hide"

  year_genre_hover = plot_year_genre.select(dict(type=HoverTool))
  year_genre_hover.tooltips = {"content": "Movie: @title Genre: @genres Grp_Year: @year_group"}
  plot_year_genre.legend.location = "top_left"
  plot_year_genre.legend.orientation = "horizontal"
  plot_year_genre.legend.click_policy="hide"

# Create two panels, one for each conference
  By_genre = Panel(child=plot_genre, title='By Genre')
  By_year = Panel(child=plot_year, title='By Year')
  By_grp_year = Panel(child=plot_year_group, title='By Grp Year')
  By_year_genre = Panel(child=plot_year_genre, title='By Year & Genre')

  # Assign the panels to Tabs
  tabs = Tabs(tabs=[By_genre, By_year, By_grp_year, By_year_genre])

  # Show the tabbed layout
  show(tabs)


# # **Predict**

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb

def logistic(trainx,trainy,testx,testy):
    logreg = LogisticRegression(fit_intercept = True, solver='liblinear').fit(trainx,trainy)
    return sum(logreg.predict(testx) == testy)/len(testy)

def LDA(trainx,trainy,testx,testy):
    LDA = LinearDiscriminantAnalysis(solver = 'lsqr').fit(trainx,trainy)
    return sum(LDA.predict(testx) == testy)/len(testy)

def KNN(trainx,trainy,testx,testy):
    neigh = KNeighborsClassifier().fit(trainx, trainy)
    return sum(neigh.predict(testx) == testy)/len(testy)

def SVM(trainx,trainy,testx,testy):
    SVM = svm.SVC().fit(trainx, trainy)
    return sum(SVM.predict(testx) == testy)/len(testy)

def balSVM(trainx,trainy,testx,testy):
    balSVM = svm.SVC(class_weight = 'balanced').fit(trainx, trainy)
    return sum(balSVM.predict(testx) == testy)/len(testy)

def XG(trainx,trainy,testx,testy):
    XGboost = xgb.XGBClassifier(objective="multi:softprob", random_state=42).fit(trainx, trainy)
    return sum(XGboost.predict(testx) == testy)/len(testy)

def NB(trainx,trainy,testx,testy):
    gnb = GaussianNB().fit(trainx, trainy)
    return sum(gnb.predict(testx) == testy)/len(testy)

def tree(trainx,trainy,testx,testy):
    tree = DecisionTreeClassifier().fit(trainx, trainy)
    return sum(tree.predict(testx) == testy)/len(testy)

def RF(trainx,trainy,testx,testy):
    rf = RandomForestClassifier().fit(trainx, trainy)
    return sum(rf.predict(testx) == testy)/len(testy)
  
def zero_rule_algorithm(trainy, testy):
    prediction = max(list(trainy),key = list(trainy).count)
    predicted = [prediction for i in range(len(testy))]
    return sum(predicted == testy)/len(testy)


# In[ ]:


len(trained_mlp_items)
dataset = pd.DataFrame(trained_mlp_items)
dataset['item_id'] = dataset.index
dataset.head(2)


# In[ ]:


dataset = pd.DataFrame(trained_mlp_items)
dataset['item_id'] = dataset.index
dataset = pd.merge(dataset, movies_info, left_on = 'item_id', right_on = 'item_id').dropna()
dataset = dataset.loc[dataset['genre_count'].isin([1])]
len(dataset)


# In[ ]:


movies_info.head(2)


# In[ ]:


# Create combine data set
# Using mlp items - implicit feedback
dataset = pd.DataFrame(trained_mlp_items)
dataset['item_id'] = dataset.index
dataset = pd.merge(dataset, movies_info, left_on = 'item_id', right_on = 'item_id').dropna()

#dataset = dataset.loc[dataset['genre_count'].isin([1])]

dataset['Label'] = pd.factorize(dataset['genres'])[0] # Create LabelEncoder
label_code_dict = dict(zip(dataset['Label'], dataset['genres'])) # Create dict to map LabelEncoder
print(len(dataset))
dataset.head()


# In[ ]:


dataset.head()


# In[ ]:


dataset = pd.DataFrame(trained_gmf_items)
len(dataset)


# In[ ]:


# Create combine data set
# Using gmf items - implicit feedback
dataset = pd.DataFrame(trained_gmf_items)
dataset['item_id'] = dataset.index
dataset = pd.merge(dataset, movies_info, left_on = 'item_id', right_on = 'item_id').dropna()

#dataset = dataset.loc[dataset['genre_count'].isin([1])]

dataset['Label'] = pd.factorize(dataset['genres'])[0] # Create LabelEncoder
label_code_dict = dict(zip(dataset['Label'], dataset['genres'])) # Create dict to map LabelEncoder
dataset.head()


# In[ ]:


dataset.to_csv("michael_data.csv")


# In[ ]:


# Define target variables
X = dataset.iloc[:, np.r_[0:64]]
Y = dataset['Label']
print("input:\n", X)
print("Target:\n", Y)


# In[ ]:


label_code_dict


# In[ ]:


dataset.head()


# In[ ]:


test = dataset[dataset['genres']== 'Horror']
test.title.value_counts()


# In[ ]:


dataset.Label.value_counts()


# In[ ]:


import matplotlib.pyplot as plt


plt.hist(Y, bins = 17)
plt.show()


# In[ ]:


# Define target variables
dataset = dataset.loc[dataset['genres'] !='Fantasy']
X = dataset.iloc[:, np.r_[0:64]]
Y = dataset['Label']
print("input:\n", X)
print("Target:\n", Y)


# In[ ]:


# k-fold
# for mlp implicit
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
balSVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}


kf = KFold(n_splits = 3, shuffle = True, random_state = 2) # split train and test using K-folds
i = 0
for train_index, test_index in kf.split(X):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    balSVM_acc.append(balSVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

# print Accuracy
print("Accuracy")
print("Logistic Regression:", sum(logistic_acc) / len(logistic_acc))
print("Naive Bayes:", sum(NB_acc) / len(NB_acc))
print("Decision Tree:", sum(tree_acc) / len(tree_acc))
print("Linear discriminant analysis:", sum(LDA_acc) / len(LDA_acc))
print("K-Neighbors Classifier:", sum(KNN_acc) / len(KNN_acc))
print("Support Vector Machines:", sum(SVM_acc) / len(SVM_acc))
print("Random forest:", sum(rf_acc) / len(rf_acc))
print("XGBoost:", sum(XG_acc) / len(XG_acc))


# In[ ]:


# k-fold
# for mlp implicit
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
balSVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}


kf = KFold(n_splits = 3, shuffle = True, random_state = 2) # split train and test using K-folds
i = 0
for train_index, test_index in kf.split(X):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    balSVM_acc.append(balSVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# for mlp implicit
from sklearn.model_selection import StratifiedShuffleSplit
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
i = 0
for train_index, test_index in sss.split(X,Y):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# for mlp implicit
# follow kfold - 3 splits

from sklearn.model_selection import StratifiedShuffleSplit
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

sss = StratifiedShuffleSplit(n_splits=3, random_state=2)
i = 0
for train_index, test_index in sss.split(X,Y):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 

results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn/46181361#:~:text=So%2C%20the%20difference%20here%20is,the%20test%20sets%20can%20overlap.

# for mlp implicit
# follow kfold - 3 spltis

from sklearn.model_selection import StratifiedKFold
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state=2) 
i = 0
for train_index, test_index in skf.split(X,Y):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn/46181361#:~:text=So%2C%20the%20difference%20here%20is,the%20test%20sets%20can%20overlap.

# FOR ALL outputs

# for mlp implicit
# follow kfold - 3 spltis

from sklearn.model_selection import StratifiedKFold
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state=2) 
i = 0
for train_index, test_index in skf.split(X,Y):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn/46181361#:~:text=So%2C%20the%20difference%20here%20is,the%20test%20sets%20can%20overlap.

# FOR ALL outputs

# for GMF implicit
# follow kfold - 3 spltis

from sklearn.model_selection import StratifiedKFold
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state=2) 
i = 0
for train_index, test_index in skf.split(X,Y):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# for gmf implicit
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
balSVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

kf = KFold(n_splits = 3, shuffle = True, random_state = 2) # split train and test using K-folds
i = 0
for train_index, test_index in kf.split(X):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    balSVM_acc.append(balSVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# for gmf implicit
# follow kfold - 3 splits

from sklearn.model_selection import StratifiedShuffleSplit
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

sss = StratifiedShuffleSplit(n_splits=3, random_state=2)
i = 0
for train_index, test_index in sss.split(X,Y):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# In[ ]:


# https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn/46181361#:~:text=So%2C%20the%20difference%20here%20is,the%20test%20sets%20can%20overlap.

# for gmf implicit
# follow kfold - 3 spltis

from sklearn.model_selection import StratifiedKFold
logistic_acc = []
LDA_acc = []
KNN_acc = []
SVM_acc = []
XG_acc = []
NB_acc = []
tree_acc = []
rf_acc = []
baseline = []
index_dict = {}

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state=2) 
i = 0
for train_index, test_index in skf.split(X,Y):
    index_dict[i] = train_index
    trainX, testX = X.iloc[train_index], X.iloc[test_index] 
    trainy, testy = Y.iloc[train_index], Y.iloc[test_index]

    # Run all models
    logistic_acc.append(logistic(trainX,trainy,testX,testy))
    LDA_acc.append(LDA(trainX,trainy,testX,testy))
    KNN_acc.append(KNN(trainX,trainy,testX,testy))
    SVM_acc.append(SVM(trainX,trainy,testX,testy))
    XG_acc.append(XG(trainX,trainy,testX,testy))
    NB_acc.append(NB(trainX,trainy,testX,testy))
    tree_acc.append(tree(trainX,trainy,testX,testy))
    rf_acc.append(RF(trainX,trainy,testX,testy))
    baseline.append(zero_rule_algorithm(trainy,testy))

    i += 1

results = pd.DataFrame(
    {'Baseline': baseline,
     'Logistic Regression': logistic_acc,
     'Linear discriminant analysis': LDA_acc,
     'Naive Bayes': NB_acc,
     'K-Neighbors Classifier': KNN_acc,
     'Decision Tree': tree_acc,
     'Random forest': rf_acc,
     'Support Vector Machines': SVM_acc,
     'XGBoost': XG_acc
    }) 
results.loc['Accuracy'] = results.mean()
results = results.transpose().sort_values(by=['Accuracy'], ascending=False)
results


# # Investigation of some movies
# 

# In[ ]:


# From the mlp item embedding, 3 genres
#Blair Witch Project, The (1999) - Horror
#Urban Legends: Final Cut (2000) - Horror

# From top 100 popular movies
#Wallace & Gromit: The Best of Aardman Animation (1996) - Animation, but in between Romance, Drama

predict_list = ['Blair Witch Project, The (1999)',
                'Urban Legends: Final Cut (2000)',
                'Wallace & Gromit: The Best of Aardman Animation (1996)']

for i in predict_list:
  target = dataset[dataset['title'] == i]
  print("Title: ", list(target['title'])[0])
  print("Genres: ", list(target['genres'])[0])
  X = target.iloc[:, np.r_[0:64]]
  pred_genre = model.predict(X)
  print("Predicted Genre: ", label_code_dict[pred_genre[0]], '\n')


# In[ ]:


American Pie (1999)
Austin Powers: The Spy Who Shagged Me (1999)


# In[ ]:


ratings.head(2)


# In[ ]:


movies.head(2)


# In[ ]:


import datetime

ratings['timestamp1'] = pd.to_datetime(ratings['timestamp'], unit = 's')
df = pd.merge(ratings, movies, how = 'left', left_on = 'movieId', right_on = 'movie_id')
df.head(2)


# In[ ]:


# is blair witch near american pie because many people watched them together?
# checked wiki - no same actors, directors. Release date were close, 14 and 9 July 1999 respectively. Watched time is mostly 2000.
# therefore, the date of release may have played a part

def movies_similarities(target_movie, neighbor_movie, ratings_movies_df):
  
  ppl_watched_targetmovie = ratings_movies_df[ratings_movies_df.title == target_movie]
  print("No of people watched target movie: ", target_movie, " ---- ", len(ppl_watched_targetmovie))
  
  list_ppl_watched = ppl_watched_targetmovie.userId.to_list()
  selected_users = ratings_movies_df[ratings_movies_df.userId.isin(list_ppl_watched)]
  
  ppl_watched_neighbormovie = selected_users[selected_users.title == neighbor_movie] # nearest movie of Blair Witch Project
  print("No of people watched neighbouring movie: ", neighbor_movie, " ---- ", len(ppl_watched_neighbormovie), '\n')

movies_similarities('Blair Witch Project, The (1999)' , 'American Pie (1999)', df)
movies_similarities('American Pie (1999)', 'Blair Witch Project, The (1999)' , df)
movies_similarities('American Pie (1999)', 'Austin Powers: The Spy Who Shagged Me (1999)', df)


# In[ ]:


movies_similarities('Urban Legends: Final Cut (2000)' , 'Cool as Ice (1991)', df)
movies_similarities('Urban Legends: Final Cut (2000)' , 'Thin Line Between Love and Hate, A (1996)', df)


#Thin Line Between Love and Hate, A (1996)


# In[ ]:


df[df['title'] == 'Cool as Ice (1991)']

