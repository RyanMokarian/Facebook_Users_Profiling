import pickle
from collections import defaultdict

from sklearn import feature_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, LinearRegression, Lasso, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from src.util import Utils
import pandas as pd
import numpy as np


def generate_age_data():
    util = Utils()
    profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
    relation_df = util.read_data_to_dataframe("../../data/Train/Relation/Relation.csv")
    merged_df = pd.merge(relation_df, profile_df, on='userid')
    merged_df['age'] = pd.cut(merged_df['age'], [0, 25, 35, 50, 200], labels=["xx-24", "25-34", "35-49", "50-xx"],
                              right=False)
    return merged_df.filter(['userid', 'like_id', 'neu'], axis=1)


def one_hot_encode(df, group_col, encode_col):
    grouped = df.groupby(group_col)[encode_col].apply(lambda lst: tuple((k, 1) for k in lst))
    category_dicts = [dict(tuples) for tuples in grouped]
    v = feature_extraction.DictVectorizer(sparse=False)
    X = v.fit_transform(category_dicts)
    one_hot = pd.DataFrame(X, columns=v.get_feature_names(), index=grouped.index)
    return one_hot


def get_one_likes_one_hot_df():
    data = generate_age_data()

    enc = OneHotEncoder(handle_unknown='ignore')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # X = X.iloc[:1000000]
    # y = y.iloc[:1000000]
    # 500,000  -> 0.7977327948181907
    # 700,000 ->  0.7775432042946905

    one_hot = one_hot_encode(X, 'userid', 'like_id')

    util = Utils()
    profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
    one_hot.index.names = ['idx']
    one_hot["userid"] = one_hot.index

    mdf = profile_df.filter(["userid", "ope", "con", "ext", "agr", "neu"], axis=1)
    merged_df = pd.merge(one_hot, mdf, on='userid')
    merged_df.drop(['userid'], 1, inplace=True)
    util.write_object_to_file(merged_df, "merged_df")
    X = merged_df[merged_df.columns[:-5]]
    Y = merged_df[merged_df.columns[-5:]]

    lasso = Lasso()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    print(np.sqrt(np.average((y_test - y_pred) ** 2, axis=0)))


if __name__ == "__main__":
    # X = merged_df.loc[:, merged_df.columns != ["ope", "con","ext","agr","neu"]]
    # Y = merged_df.loc[:, merged_df.columns == ["ope", "con","ext","agr","neu"]]
    # get_one_likes_one_hot_df()
    util = Utils()
    df = util.read_pickle_from_file("merged_df")
    # df = util.apply_fast_ica(df, 1000,5)
    X = df[df.columns[:-5]]
    Y = df[df.columns[-5:]]

    lasso = RidgeCV()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    print(np.sqrt(np.average((y_test - y_pred) ** 2, axis=0)))

    gooz = ""
