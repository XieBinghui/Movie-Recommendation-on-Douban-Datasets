import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import random
import datetime
from myModel import *
import tensorflow as tf


data_movies = pd.read_csv('movies.csv')
data_user = pd.read_csv('user.csv')

for i in range(len(data_movies)):
    for key in range(1, data_movies.columns.size):
        data_movies.iloc[i, key] = data_movies.iloc[i, key][2:len(data_movies.iloc[i, key])-3]
for i in range(1, len(data_user)):
    for key in range(2, data_user.columns.size):
        data_user.iloc[i, key] = data_user.iloc[i, key][2:len(data_user.iloc[i, key])-3]
movie_id_to_name = pd.Series(data_movies.title.values,index=data_movies.id.values).to_dict()
movie_name_to_id = pd.Series(data_movies.id.values,index=data_movies.title.values).to_dict()


data_user_train,data_user_test = train_test_split(data_user,\
                                                 random_state = 1,test_size = 0.20)

print("Number of training data:"+str(len(data_user_train)))
print("Number of test data:"+str(len(data_user_test)))

def rating_splitter_by_one_user(data):
    data = data[1:len(data)-2]
    like_movie_id = []
    dislike_movie_id = []
    for ele in data.split(","):
        num = re.findall('\d+', ele)
        if len(num)>1:
            if int(num[1]) >= 4:
                like_movie_id.append(num[0])
            else: dislike_movie_id.append(num[0])
    return like_movie_id,dislike_movie_id


splitted_movies = []
for rate in data_user_train['rates']:
    a,b = rating_splitter_by_one_user(rate)
    splitted_movies.append(a)
    splitted_movies.append(b)


for movie_list in splitted_movies:
    random.shuffle(movie_list)


def search_movie(movie_name):
    if movie_name in movie_name_to_id.keys():
        return movie_name_to_id[movie_name]
    else:
        print("Invalid movie_name")
        return


def recommender(movie_name, negative=None, topn=5):
    movie_id = search_movie(movie_name)
    recommend_movie_ls = []
    if movie_id not in model.movies_list:
        print("Invalid movie_name")
        return
    model.eval(movie_id, N=5)

start = datetime.datetime.now()
#print(splitted_movies)

with tf.Graph().as_default(), tf.Session() as session:
    model = Model(session, data=splitted_movies, epoch=10, embed_dim=30, negatives=5,
                       learning_rate=1e-3, batch_size=256, save_path='./model')
    # recommender(movie_name='星际迷航3：超越星辰', topn=5)

print("Time passed:" + str(datetime.datetime.now()-start))
# model_item2vec.save('item2vec_20190501')
# model = Word2Vec.load("item2vec_20190501")



