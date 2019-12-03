import pickle

from src.util import Utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import array
from scipy.sparse import csr_matrix
from collections import defaultdict


class LikeClassifier:

    @staticmethod
    def generate_gender_data():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        relation_df = util.read_data_to_dataframe("../../data/Train/Relation/Relation.csv")
        merged_df = pd.merge(relation_df, profile_df, on='userid')
        return merged_df.filter(['like_id', 'gender'], axis=1)


    @staticmethod
    def generate_personality_data():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        relation_df = util.read_data_to_dataframe("../../data/Train/Relation/Relation.csv")
        # I had to uncomment the below line as I got a memory error to read 1,671,354 lines of the Relation.csv file.
        # relation_df = relation_df[:50000]
                    # Note: below is an easy way to create a likes feature matrix using pivot libary
                    # relation_df.insert(3, "likes_onehot", [1] * len(relation_df.index), True)
                    # relation_df_onehot = pd.pivot(relation_df, index='userid', columns='like_id', values='likes_onehot')
                    # relation_df_onehot = relation_df_onehot.fillna(0)  # replace NaN with zeros
        # Creation of a likes feature matrix without using above pivot library:
        # 1. create a set from likes and a dictionary (keys: users; values: list of likes)
        user_id_set = relation_df['userid'].unique().tolist()
        like_id_set = relation_df['like_id'].unique().tolist()
        user_like_dict = defaultdict(list)
        for index, rows in relation_df.iterrows():
            if rows.userid in user_like_dict:
                user_like_dict[rows.userid].append(like_id_set.index(rows.like_id)) # np.where(like_id_set == rows.like_id)
            else:
                user_like_dict[rows.userid] = [like_id_set.index(rows.like_id)] # like_id_set.index(rows.like_id)
        # up to here I had no memory problem for making a dictionary from the whole data (1,671,354 lines)

        # 2. make a 2d array from the created dictionary
        n_user_id = len(user_id_set)
        n_like_id = len(like_id_set)
        user_like_array = np.zeros([n_user_id, n_like_id], dtype = np.int8) # I got a memory problem for making (25000*536204) matrix
        for i in range(n_user_id):
            for j in range(0, n_like_id):
                if j in user_like_dict[user_id_set[i]]:
                    user_like_array[i, j] = 1
        # 3. convert 2d array to df
        user_like_df = pd.DataFrame(user_like_array)
        user_like_df.insert(0, "userid", user_id_set, True)
        merged_df = pd.merge(user_like_df, profile_df, on='userid')
            #     #In case conversion to a sparse matrix required, the code is:
                #     merged_df_1 = csr_matrix(np.asarray(merged_df))   and to return it to the original, use .todense()
        return merged_df


    @staticmethod
    def generate_age_data():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        relation_df = util.read_data_to_dataframe("../../data/Train/Relation/Relation.csv")
        merged_df = pd.merge(relation_df, profile_df, on='userid')
        return merged_df.filter(['like_id', 'age'], axis=1)

    @staticmethod
    def categorical_convertion(merged_df):
        merged_df['ope_catg'] = pd.cut(merged_df['ope'], bins=[1, 2, 3, 4, 5], labels=['1.5', '2.5', '3.5', '4.5'],
                                       include_lowest=True)
        merged_df['con_catg'] = pd.cut(merged_df['con'], bins=[1, 2, 3, 4, 5], labels=['1.5', '2.5', '3.5', '4.5'],
                                       include_lowest=True)
        merged_df['ext_catg'] = pd.cut(merged_df['ext'], bins=[1, 2, 3, 4, 5], labels=['1.5', '2.5', '3.5', '4.5'],
                                       include_lowest=True)
        merged_df['agr_catg'] = pd.cut(merged_df['agr'], bins=[1, 2, 3, 4, 5], labels=['1.5', '2.5', '3.5', '4.5'],
                                       include_lowest=True)
        merged_df['neu_catg'] = pd.cut(merged_df['neu'], bins=[1, 2, 3, 4, 5], labels=['1.5', '2.5', '3.5', '4.5'],
                                       include_lowest=True)

    @staticmethod
    def split_data(df):
        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test


def variable_predictor(df, predicted_variable):
    X_train, X_test, y_train, y_test = LikeClassifier.split_data(df)
    neigh = KNeighborsClassifier(n_neighbors=6)
    neigh.fit(X_train, y_train)
    pickle.dump(neigh, open("../resources/KNNlikes_" + predicted_variable + ".sav", 'wb'))
    y_pred = neigh.predict(X_test)
    print("KNN acc: ", accuracy_score(y_test, y_pred))

def variable_predictor_linreg(df, predicted_variable):
    X_train, X_test, y_train, y_test = LikeClassifier.split_data(df)
    # clf = GradientBoostingRegressor(n_estimators=300)
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train, y_train)
    # pickle.dump(clf, open("../resources/LinReglikes_" + predicted_variable + ".sav", 'wb'))
    y_pred = clf.predict(X_test)
    print('predicted Personality Trait ', predicted_variable, ':\n')
    print('Slope of the regression line: ', clf.coef_)
    print('Intercept of the regression line', clf.intercept_)
    print("R-squared for Train: %.2f" % clf.score(X_train, y_train))
    print("R-squared for Test: %.2f" % clf.score(X_test, y_test)) # 1 means 100% accurate prediction
    rms = sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE = ', rms)
    pass


if __name__ == "__main__":
    # df_gender = LikeClassifier().generate_gender_data()
    # X_train, X_test, y_train, y_test = LikeClassifier.split_data(df_gender)
    # variable_predictor(df_gender, 'gender')

    # df_age = LikeClassifier().generate_age_data()
    # df_age['age'] = pd.cut(df_age['age'], [0, 25, 35, 50, 200], labels=["xx-24", "25-34", "35-49", "50-xx"],
    #                        right=False)
    # variable_predictor(df_age, 'age-group')

    # Personality Traits
    merged_df = LikeClassifier().generate_personality_data()
    cols = [0, -8, -7, -6, -5, -4, -2, -1]
    df_ext = merged_df.drop(merged_df.columns[cols], axis=1) # here df_ext includes all like features and ext column
    variable_predictor_linreg(df_ext, 'ext')