from src.util import Utils
import pandas as pd
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# values = np.asarray (data)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# # invert first example
# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# print(inverted)
util = Utils()
profile_df = util.read_data_to_dataframe("./data/Train/Profile/Profile.csv")
relation_df = util.read_data_to_dataframe("./data/Train/Relation/Relation.csv")
relation_df = relation_df[:2000] # with 100,000 worked but with 200,000 I got a low memory error. dara has 1671353 rows
relation_df.insert(3, "likes_onehot", [1] * len(relation_df.index) , True)
relation_df_onehot = pd.pivot(relation_df, index='userid', columns='like_id', values='likes_onehot')
relation_df_onehot = relation_df_onehot.fillna(0)# replace NaN with zeros
merged_df = pd.merge(relation_df_onehot, profile_df, on='userid')


print('hi')