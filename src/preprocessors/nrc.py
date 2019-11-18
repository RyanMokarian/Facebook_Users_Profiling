from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

from src.util import Utils

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_data():
    util = Utils()
    profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
    nrc_df = util.read_data_to_dataframe("../../data/Train/Text/nrc.csv")
    nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
    merged_df = pd.merge(nrc_df, profile_df, on='userid')
    return merged_df.filter(['positive', 'negative', 'anger', 'anticipation', 'disgust',
                             'fear', 'joy', 'sadness', 'surprise', 'trust', 'ope', 'con', 'ext', 'agr', 'neu'], axis=1)


if __name__ == '__main__':
    df = get_data()

    # df.drop(['negative', 'anticipation', 'fear', 'joy', 'sadness'], axis=1, inplace=True)
    X = df.iloc[:, 0:-5]  # independent columns
    print(X.describe())
    X = np.log(X + 1)
    X=(X-X.min())/(X.max()-X.min())
    X.hist()
    plt.show()
    print(X.describe())
    y = df[df.columns[-5:]]

    reg = LinearRegression()

    reg.fit(X, y)
    pickle.dump(reg, open("../resources/LinearReg_Personality.sav", 'wb'))
    # print(np.sqrt(np.average((y_test - predict) ** 2, axis=0)))