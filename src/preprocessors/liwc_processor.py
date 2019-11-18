import matplotlib
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from src.util import Utils
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm



class LiwcProcessor:
    @staticmethod
    def read_data():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        liwc_df = util.read_data_to_dataframe("../../data/Train/Text/liwc.csv")
        liwc_df.rename(columns={'userId': 'userid'}, inplace=True)
        merged_df = pd.merge(profile_df, liwc_df, on='userid', how='left')
        return merged_df.filter(['WC','WPS','Sixltr','Dic','Numerals','funct','pronoun','ppron','i','we','you','shehe',
                                 'they','ipron','article','verb','auxverb','past','present','future','adverb','preps',
                                 'conj','negate','quant','number','swear','social','family','friend','humans','affect',
                                 'posemo','negemo','anx','anger','sad','cogmech','insight','cause','discrep','tentat',
                                 'certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health',
                                 'sexual','ingest','relativ','motion','space','time','work','achieve','leisure','home',
                                 'money','relig','death','assent','nonfl','filler','Period','Comma','Colon','SemiC',
                                 'QMark','Exclam','Dash','Quote','Apostro','Parenth','OtherP','AllPct', 'age'], axis=1)

    @staticmethod
    def hist_features(df):
        for col in df.columns:
            sns.distplot(df[col], kde=False)
            plt.show()


def remove_outliers(df):
    for col in df.columns:
        data = df[col]
        upper_lim = data.mean() + data.std() * 3
        lower_lim = data.mean() - data.std() * 3
        df[col] = data[(data < upper_lim) & (data > lower_lim)]


def transform(df):
    x = df.values
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(df)
    return pd.DataFrame(x_scaled)


def plot_pca(df):
    normalized_df = transform(df.iloc[:, 0: -1])
    #12 they, 28 family
    pca = PCA(2)
    projected = pca.fit_transform(normalized_df, df.age)
    colors = ['red', 'green', 'blue', 'yellow']
    plt.scatter(normalized_df.iloc[:, 12], normalized_df.iloc[:, 28],
           c=df.age, edgecolor='none', alpha=0.5, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()


def hist(df):
    x0_24 = df.loc[df.iloc[:, -1] == 0]
    x25_34 = df.loc[df.iloc[:, -1] == 1]
    x35_49 = df.loc[df.iloc[:, -1] == 2]
    x50_xx = df.loc[df.iloc[:, -1] == 3]

    for i in range(x0_24.shape[1] - 1):
        plt.hist([x0_24.iloc[:, i], x25_34.iloc[:, i], x35_49.iloc[:, i], x50_xx.iloc[:, i]],
                 color=['orange', 'lime', 'red', 'blue'], density=True)
        plt.legend(['0_24', '25_34', '35_49', '50_xx'])
        plt.ylabel(list(df)[i])
        plt.show()


df = LiwcProcessor.read_data()
# Remove outliers
df = df[df['age'] <= 100]
df_age = pd.cut(df['age'], [0, 25, 35, 50, 200], labels=[0, 1, 2, 3], right=False)
df['age'] = df_age

# Drop features that dont discriminate
df = df.drop(['number', 'Numerals','AllPct', 'OtherP', 'Parenth', 'Quote', 'Dash', 'Exclam',
              'SemiC', 'Colon', 'Comma', 'Period', 'QMark'], axis=1)
hist(df)

df = df.drop(['age'], axis=1)

# tranformations
tranformed_df = transform(df)
# remove_outliers(new_df)

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(tranformed_df, df_age)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(df.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(10,'Score'))

# print(df.info())
# print(df.describe())
# LiwcProcessor.plot_features(df)
gooz =""