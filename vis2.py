# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:31:43 2019

@author: karapet
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:24:16 2019

@author: Balint
"""

import numpy as np
import pandas as pd

df = pd.read_csv("movie_metadata.csv")

#%%
df.head()
#%%
pd.set_option("max_columns",None)

#%%
df.head()

#%%

df.describe()


#%%% HIANYZO ERTETEK

print(df[df.isnull().any(axis=1)])

#%%
len(df)

#%%
# eldobjuk a NAN-t tartalmazo sorokat
df = df[~df.isnull().any(axis=1)]

#%%
len(df)

#%%
df.head()
# genres...
feats_cat=["color","director_name","actor_1_name","actor_2_name","actor_3_name","language","country","content_rating"]
# figyelem: aspect_ratio nem igazi numerikus  valtozo
feats_num=['num_critic_for_reviews', 'duration', 'director_facebook_likes',
       'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
       'num_user_for_reviews', 'budget', 'title_year',
       'actor_2_facebook_likes',  'aspect_ratio',
       'movie_facebook_likes']
target = ['imdb_score']

#%%
len(df)
#%%
feat="language"
df[feat].nunique()
#%%
df[feat].value_counts()
#%%
df[feat].unique()

#%%
feat_value="English"
df[feat]==feat_value
#%%
df[df[feat]==feat_value]
#%%
### KATEGORIKUS VALTOZOK FELDOLGOZASA
df['country'].unique()
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

df = pd.get_dummies(df,columns=['color','language','country','content_rating'])
#%%
df.columns
#%%
print(df['genres'])

#%%
"hello|world|123".split("|")
#%%
genres = pd.Series()
for value in df['genres']:
    genres = genres.append(pd.Series(value.split("|")), ignore_index=True)
print(len(genres))
#%%
genres = genres.drop_duplicates()
print(len(genres))
#%%
genres
#%%
print(genres.reset_index(drop=True))
#%%
for feat in genres:
    df[feat]=0
    df[feat]=df['genres'].apply(lambda x: 1 if feat in str(x) else 0, 1)
#%%
print(df.columns)
#%%
print(df.head())

#%%
######### OUTLIEREK KERESES
k=5
for feat in feats_num:
    print("outlierek szama a %s featureben: %d / %d)" % (feat, \
                             len(df[ abs(df[feat]-df[feat].mean())>k*df[feat].std() ]), \
                             len(df)))
    
    mean_feat_min = np.mean(df[feat]) - np.std(df[feat])*k #df[ abs(df[feat]-df[feat].mean())>k*df[feat].std() ]
    mean_feat_max = np.mean(df[feat]) + np.std(df[feat])*k #df[ abs(df[feat]-df[feat].mean())>k*df[feat].std() ]
    df[feat] = df[feat].apply(lambda x: mean_feat_min if x<mean_feat_min else x,1)
    df[feat] = df[feat].apply(lambda x: mean_feat_max if x>mean_feat_max else x,1)
    
    #print(df[ abs(df[feat]-df[feat].mean())>k*df[feat].std() ][feat])
#%%
#%%
feat="director_facebook_likes"
print(np.std(df[feat]))
print(np.mean(df[feat]))
#%%
import seaborn as sns
sns.distplot(df['director_facebook_likes'], kde=False)
#%%
#%%
for feat in feats_num:
    print("outlierek szama a %s featureben: %d / %d)" % (feat, \
                             len(df[ abs(df[feat]-df[feat].mean())>k*df[feat].std() ]), \
                             len(df)))
#%%
del df['genres'] 
del df['movie_imdb_link']
del df['plot_keywords']
del df['actor_3_name']
del df['actor_2_name']
del df['actor_1_name']
del df['movie_title']
del df['director_name']

#%%
df.to_hdf('processed_data.hdf5',key="imdb")
df = pd.read_hdf('processed_data.hdf5', key='imdb')

#%%
X = df.loc[:,df.columns != 'imdb_score'].values
Y = df['imdb_score']
#%%
print(X.shape)
print(Y.shape)
#%%
#### LINEARIS REGRESSZIO
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=42)
#%%
linreg = LinearRegression()
linreg.fit(X_train, Y_train)
preds = linreg.predict(X_test)
print("MSE:", mean_squared_error(Y_test, preds))
#%%
pred_mean=np.mean(Y_train)
#%%
print("MSE mean:", mean_squared_error(Y_test,pred_mean*np.ones(len(Y_test))))
#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

sns.regplot(Y_test, preds).set(xlim=(1,10),ylim=(1,10))
#%%%
df_linreg = pd.DataFrame([feat for feat in df.columns[df.columns != 'imdb_score']], columns=['features'])
df_linreg['coeffs']=linreg.coef_

print(df_linreg.sort_values(by='coeffs'))
#%%
linreg.coef_

#%%
linreg.intercept_
