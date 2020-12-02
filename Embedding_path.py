#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import manhattan_distances

# Define number of movies before and after stepping stones
n_front = 10
n_back = 10

all_genres = ratings_movies[['movieId', 'title', 'genres']]
all_genres = all_genres.drop_duplicates()
all_genres['genre'] = all_genres.genres.str.split('|')

for type_genre in unique_genres:

  sub_df = all_genres[all_genres.genres.str.contains(type_genre)]

  all_users_targetmovies = pd.DataFrame()
  selected_movies = list(set(sub_df['movieId']))
  print('genre: ', type_genre, ',    number of movies with selected genre: ', len(selected_movies))

  all_users_targetmovies = pd.DataFrame()

  for target in selected_movies:

    # chosee only instances where the selected movies has at least 'n_front' movies before them (improve computational effiency)
    filter_movies = ratings_movies[(ratings_movies['movieId'] == target) & (ratings_movies['movies_order'] > n_front) ] # Selected movies 
    movies_name = list(ratings_movies[ratings_movies['movieId'] == target]['title'].drop_duplicates())[0]

    # to ignore blank dataframes
    if len(filter_movies) != 0:
  
      # Check when user watch selected movies
      order_dict = {}
      for i in list(filter_movies.index.values):
        order_dict[filter_movies['userId'][i]] = filter_movies['movies_order'][i]

      seq_df = pd.DataFrame()
      for k in list(order_dict.keys()):
        df_temp = ratings_movies.loc[(ratings_movies['userId'] == k) & ((order_dict[k] - n_front) <= ratings_movies['movies_order']) & ((order_dict[k] + n_back) >= ratings_movies['movies_order'])]
        seq_df = pd.concat([seq_df, df_temp])

      seq_df['count'] = 1
      seq_df['new_movies_order'] = seq_df.groupby(by = ['userId'])['count'].transform(lambda x: x.cumsum())

      genres_seq_first = seq_df.groupby('userId')['genre'].apply(lambda x: x.tolist()) # Get sequence of genres by user
 
      user_path_df = pd.DataFrame()
      for user in list(genres_seq_first.index.values):
        check_previous = 0
        num_movies = 0
        for j in genres_seq_first[user][:n_front]: # check previous movies not in selected genre
          if type_genre not in j:
            check_previous += 1
        if check_previous == n_front: #and num_movies >= n_after: 
          user_path_df = pd.concat([user_path_df, seq_df[seq_df['userId'] == user][['userId','movieId','title','genre','timestamp', 'new_movies_order']]])
  
      # Next steps - using the user_path_df of the target movie, find all users' previous movie cluster and target movie cluster and distance
    
      if len(user_path_df) != 0: 
        all_user_path_df = pd.merge(user_path_df, dataset_cluster_label, how = 'left', on = 'movieId')

        na_users = all_user_path_df[all_user_path_df.isnull().any(axis=1)].userId
        all_users = all_user_path_df.userId.unique()
        filtered_users = set(all_users) - set(na_users)

        stepping_movies = []
        previousmovie_cluster_label_list = []
        targetmovie_cluster_label_list = []
        linear_distance_list = []
        nonlinear_distance_list = []
        num_movies = []
        users_list = []
    

        for user_targetmovie in filtered_users:
          df = all_user_path_df[all_user_path_df['userId'] == user_targetmovie]
          movie_title = df.iloc[n_front: n_front +1,]['title_y'].values.tolist()
          previous_movie_label = df.iloc[n_front -1:n_front,]['cluster_label'].values.tolist()
          target_movie_label = df.iloc[n_front:n_front+1,]['cluster_label'].values.tolist()
          #previous_movie_label = df.iloc[n_front:n_front+1,]['distance_centroid'].values.tolist()
          #target_movie_distance_centroid = df.iloc[n_front:n_front+1,]['distance_centroid'].values.tolist()

          users_list.append(user_targetmovie)
          stepping_movies.extend(movie_title)
          previousmovie_cluster_label_list.extend(previous_movie_label)
          targetmovie_cluster_label_list.extend(target_movie_label)

          # find distance
          target_movie_location = df[df['userId']== user_targetmovie][n_front: n_front + 1][np.r_[0:64]].to_numpy() 
          previous_movie_location = df[df['userId'] == user_targetmovie][n_front-1 : n_front][np.r_[0:64]].to_numpy()
          linear_distance_list.append(np.linalg.norm(target_movie_location - previous_movie_location))
          nonlinear_distance_list.extend(manhattan_distances(target_movie_location, previous_movie_location).ravel())

          genres_seq = df.groupby('userId')['genre'].apply(lambda x: x.tolist()) # Get sequence of genres by user

          count = 0
          for genres_list in genres_seq[user_targetmovie][n_front:]: 
            if type_genre in genres_list:
              count += 1
          num_movies.append(count)

        final_df = pd.DataFrame({'user': users_list, 'target_movie_cluster': targetmovie_cluster_label_list, 'previous_movie_cluster':previousmovie_cluster_label_list,
                                 'target_movie_title':stepping_movies, 'linear_distance': linear_distance_list, 'nonlinear_distance': nonlinear_distance_list, 'movies_count': num_movies})
    
        all_users_targetmovies = pd.concat([all_users_targetmovies, final_df])

  new_df = all_users_targetmovies.drop_duplicates()

  print('all users: ', len(all_users_targetmovies), ' validation check on no duplicates: ', len(new_df))

  new_df2 = pd.merge(new_df, centroids_df, how = 'left', left_on = 'target_movie_title', right_on = 'title')

  # use the above condition to filter out the corresponding movie_cluster
  new_df2['distance_lastmovie_centroid'] = new_df2.apply (lambda row: centroid_dis(row), axis=1)
  new_df2['user_count'] = 1
  analysis = new_df2[['user', 'target_movie_cluster', 'previous_movie_cluster', 'target_movie_title', 'linear_distance', 'nonlinear_distance', 'movies_count', 'user_count',
                            'distance_centroid', 'distance_lastmovie_centroid']]
  users_in_genre = analysis.pivot_table(values = ['movies_count','user_count'], index = ['target_movie_title', 'target_movie_cluster', 'previous_movie_cluster'],
                                              aggfunc = {'movies_count': np.sum, 'user_count' : np.sum}).reset_index()
        
  centroids_dis = analysis[['target_movie_cluster', 'previous_movie_cluster', 'target_movie_title', 'distance_centroid', 'distance_lastmovie_centroid']]
  centroids_dis = centroids_dis.drop_duplicates()

  testing = analysis[['target_movie_cluster', 'previous_movie_cluster', 'target_movie_title', 'linear_distance', 'movies_count', 'nonlinear_distance']]
  testing_corr = testing.groupby(['target_movie_cluster', 'previous_movie_cluster', 'target_movie_title']).corr().reset_index()
  testing_corr = testing_corr[testing_corr['level_3'] == 'movies_count'].drop(['movies_count'], axis = 1)

  final_analysis1 = pd.merge(testing_corr, users_in_genre, how = 'left', on = ['target_movie_title', 'target_movie_cluster', 'previous_movie_cluster'])
  final_analysis2 = pd.merge(final_analysis1, centroids_dis, how = 'left', on = ['target_movie_title', 'target_movie_cluster', 'previous_movie_cluster'])
  filter_users = final_analysis2[final_analysis2['user_count'] >= 50] # pvalue
  filter_users['avg_movies'] = filter_users['movies_count'] / filter_users['user_count']
  print('final df with number of datapoints: ', len(filter_users))

  if len(filter_users) >= 2:
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15,8))

    corr_centroid, pvalue_centroid = pearsonr(filter_users['distance_centroid'], filter_users['avg_movies'])
    corr_priorcentroid__, pvalue_priorcentroid = pearsonr(filter_users['distance_lastmovie_centroid'], filter_users['avg_movies'])

    sns.regplot(filter_users['distance_centroid'], filter_users['avg_movies'], ax = ax3, order = 1, logx = False)
    sns.regplot(filter_users['distance_lastmovie_centroid'], filter_users['avg_movies'], ax = ax4)

    plt.suptitle("Movies count vs other variables - %s genre" %type_genre)
    ax3.set_title("Dist to centroid vs total movie_count \n corr: %f  pvalue: %f" % (corr_centroid, pvalue_centroid))
    ax4.set_title("Dist previous movie's centroid vs total movie_count - \n corr: %f  pvalue: %f" % (corr_priorcentroid__, pvalue_priorcentroid))
    plt.show()
    fig.savefig('%s.png' %type_genre)
        
  else:
    print('%s has less than 2 datapoints' %type_genre, '\n')

