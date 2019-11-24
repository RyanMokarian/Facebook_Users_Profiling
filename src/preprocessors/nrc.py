import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.util import Utils

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_data():
    util = Utils()
    profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
    profile_df= profile_df.filter(['userid', 'ope', 'con', 'ext', 'agr', 'neu'], axis=1)

    nrc_df = util.read_data_to_dataframe("../../data/Train/Text/nrc.csv")
    liwc_df = util.read_data_to_dataframe("../../data/Train/Text/liwc.csv")
    nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
    liwc_df.rename(columns={'userId': 'userid'}, inplace=True)
    merged_df = pd.merge(nrc_df, liwc_df, on='userid')
    merged_df = pd.merge(merged_df, profile_df, on='userid')
    merged_df.drop(['userid'], axis=1, inplace=True)
    return  merged_df


if __name__ == '__main__':
    """
    Ran on 24, Nov : [0.61674547 0.69935273 0.80564432 0.65389657 0.7969793 ]
    """
    df = get_data()

    # df.drop(['negative', 'anticipation', 'fear', 'joy', 'sadness'], axis=1, inplace=True)
    X = df.iloc[:, 0:-5]  # independent columns
    print(X.describe())
    X = np.log(X + 1)
    X = (X - X.min()) / (X.max() - X.min())
    X.hist()
    plt.show()
    print(X.describe())
    y = df[df.columns[-5:]]
    util  =Utils()


    reg = LinearRegression()
    # reg.fit(X, y)
    # pickle.dump(reg, open("../resources/LinearReg_Personality.sav", 'wb'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    print(np.sqrt(np.average((y_test - predict) ** 2, axis=0)))
