# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:22:28 2019

@author: karapet
"""

import numpy as np
import pandas as pd

df = pd.read_csv("movie_metadata.csv")

pd.set_option("max_columns", None)

#%%
print(df.tail(1))

#%%
print(df.describe())

#%%% hianyzo ertekek keresese

print(df[df.isnull().any(axis=1)]) #soronkent kiirja, 1287 sor van ahol hianyzik
#%%
print(len(df))

df = df[~df.isnull().any(axis=1)]

print(len(df))

#%%
print(df.columns)


#%%

feats_cat = ["color","director_name","actor_1_name","actor_2_name","actor_3_name","language","country","content_rating"]

#%%
feats_num = ['num_critic_for_reviews', 
                    'duration',
                    'director_facebook_likes', 
                    'actor_3_facebook_likes', 
                    'actor_1_facebook_likes', 
                    'gross', 
                    'num_voted_users', 
                    'cast_total_facebook_likes',
                    'facenumber_in_poster', 
                    'num_user_for_reviews', 
                    'budget',
                    'title_year', 
                    'actor_2_facebook_likes', 
                    'aspect_ratio',
                    'movie_facebook_likes',
                   ]

target = ["imdb_score"]

#%%




for feat in feats_cat:
    print(feat, "feldolgozasa, egyedi ertekek")
    if df[feat].nunique()<50:
        print(df[feat].value_counts())
        for feat_value in df[feat].unique():
            if (len(df[df[feat]==feat_value])/len(df)<=0.05):
                df[feat] = df[feat].apply(lambda x: "OTHER" if x==feat_value else x,1)
    print("\n\n")
    

#%%
df['country'].value_counts()

#%%
df = pd.get_dummies(df,columns=[''])

df.columns


#%%
print(df['genres'])
#%%
genres = pd.Series()

for value in df['genres']:
    genres = genres.append(pd.Series(value.split("|")), ignore_index=True)

genres = genres.drop_duplicates()
print(len(genres))
print(genres.reset_index())

for feat in genres:
    df[feat] = 0
    df[feat] = df['genres'].apply(lambd x: 1 if feat in str(x) else 0,1) #,1 az az hogy soronkent menjen vegig a fv

#%%
print(df.columns)
print(genres)



#%%

for feats in feats_num:
    print ("outlierek szama a %s featureben: %d / %d" % (feat, \
           len(df[ absfeat]-df[feat].mean())


#%%
df.to_hdf('procesased_data.hdf5',key='imdb')
df = pd.read_hdf('procesased_data.hdf5',key='imdb')

X = df.loc[:,df.columns != 'imdb_score'].values
Y = df['imdb_score']

#%%
print(X.shape)
print(Y.shape)



#%%

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=42)

linreg = LinearRegression()
linreg.fit(X_train,Y_train)
preds = linreg.predict(X_test)
print("MSE:", mean_squared_error(Y_test,preds))


