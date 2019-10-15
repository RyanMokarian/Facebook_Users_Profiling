import pickle

from src.util import Utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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
        merged_df = pd.merge(relation_df, profile_df, on='userid')
        LikeClassifier.categorical_convertion(merged_df)
        ope = merged_df.filter(['like_id', 'ope_catg'], axis=1)
        con = merged_df.filter(['like_id', 'con_catg'], axis=1)
        ext = merged_df.filter(['like_id', 'ext_catg'], axis=1)
        age = merged_df.filter(['like_id', 'agr_catg'], axis=1)
        neu = merged_df.filter(['like_id', 'neu_catg'], axis=1)
        return ope, con, ext, age, neu

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


def variable_predictor(df,predicted_variable):
    X_train, X_test, y_train, y_test = LikeClassifier.split_data(df)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    pickle.dump(neigh, open("../resources/KNNlikes_"+predicted_variable+".sav", 'wb'))
    y_pred = neigh.predict(X_test)
    print("KNN acc: ", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    df_gender = LikeClassifier().generate_gender_data()
    X_train, X_test, y_train, y_test = LikeClassifier.split_data(df_gender)
    variable_predictor(df_gender, 'gender')

    df_ope, df_con, df_ext, df_agr, df_neu = LikeClassifier().generate_personality_data()
    variable_predictor(df_ope,'ope')
    variable_predictor(df_con,'con')
    variable_predictor(df_ext,'ext')
    variable_predictor(df_agr,'agr')
    variable_predictor(df_neu,'neu')

    df_age = LikeClassifier().generate_age_data()
    df_age['age'] = pd.cut(df_age['age'], [0, 25, 35, 50, 200], labels=["xx-24", "25-34", "35-49", "50-xx"],
                           right=False)
    variable_predictor(df_age, 'age-group')