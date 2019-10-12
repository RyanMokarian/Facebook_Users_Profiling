import pickle

from src.util import Utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class LikeClassifier:

    @staticmethod
    def generate_data():
        util = Utils()
        relation_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        profile_df = util.read_data_to_dataframe("../../data/Train/Relation/Relation.csv")
        merged_df = pd.merge(relation_df, profile_df, on='userid')
        return merged_df.filter(['like_id', 'gender'], axis=1)

    @staticmethod
    def split_data(df):
        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = LikeClassifier().generate_data()
    X_train, X_test, y_train, y_test = LikeClassifier.split_data(df)

    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(X_train, y_train)
    pickle.dump(clf, open("../resources/SGDlikes.sav", 'wb'))
    y_pred = clf.predict(X_test)
    print("SDG acc: ", accuracy_score(y_test, y_pred))

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    pickle.dump(neigh, open("../resources/KNNlikes.sav", 'wb'))
    y_pred = neigh.predict(X_test)
    print("KNN acc: ", accuracy_score(y_test, y_pred))