from scipy.stats import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC

from src.util import Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
                             'fear', 'joy', 'sadness', 'surprise', 'trust', 'ope'], axis=1)

if __name__ == '__main__':
    df = get_data()

    df.drop(['negative', 'anticipation', 'fear', 'joy', 'sadness'], axis=1, inplace=True)
    X = df.iloc[:, 0:-1]  # independent columns
    print(X.describe())
    X = np.log(X + 1)
    X=(X-X.min())/(X.max()-X.min())
    X.hist()
    plt.show()
    print(X.describe())
    y = df.iloc[:, -1]  # target column i.e price range

    import statsmodels.api as sm
    from scipy import stats

    mod = sm.OLS(y, X)
    fii = mod.fit()
    print(fii.summary2())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg = LinearRegression()

    # scores = cross_val_score(reg, X, y, cv=5)
    reg.fit(X_train, y_train)
    predict = reg.predict(X_test)
    print(np.sqrt(np.average((y_test-predict)**2))) # this is their method of error calculation

    # This will print the mean of the list of errors that were output and
    # provide your metric for evaluation