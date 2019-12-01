from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.util import Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NRC_Classifier:
    @staticmethod
    def get_nrc_personality_training_data():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        nrc_df = util.read_data_to_dataframe("../../data/Train/Text/nrc.csv")
        nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
        merged_df = pd.merge(nrc_df, profile_df, on='userid')
        return merged_df

    @staticmethod
    def plot_personality_histograms(df_personality):
        data = df_personality.to_numpy()
        for i in range(5):
            x = data[:,i]
            plt.hist(x, density=True, bins=10)
            plt.xlabel(df_personality.columns[i])
            plt.ylabel('Probability')
            plt.show()

    def radar_chart(labels, values, title):
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_title(title)
        ax.grid(True)
        plt.show()

    @staticmethod
    def split_data(df):
        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test


    def variable_predictor_linreg(df_x, df_y, predicted_variable):
        df = pd.concat([df_x, df_y], axis=1)
        X_train, X_test, y_train, y_test = NRC_Classifier.split_data(df)
        # clf = GradientBoostingRegressor(n_estimators=300)
        clf = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5)
        clf.fit(X_train, y_train)
        # pickle.dump(clf, open("../resources/LinReglikes_" + predicted_variable + ".sav", 'wb'))
        y_pred = clf.predict(X_test)
        print('Predicted Variable: ', predicted_variable)
        print('Slope of the regression line: ', clf.coef_)
        print('Intercept of the regression line', clf.intercept_)
        print("R-squared for Train: %.2f" % clf.score(X_train, y_train))
        print("R-squared for Test: %.2f" % clf.score(X_test, y_test))# 1 means 100% accurate prediction
        rms = sqrt(mean_squared_error(y_test, y_pred))
        print('RMSE = ', rms)


if __name__ == "__main__":

    nrc_profile_df = NRC_Classifier().get_nrc_personality_training_data()
    new_data = nrc_profile_df.dropna(axis=0, how='any')
    print("Length of the original data frame:", len(nrc_profile_df))
    print("Length of the data frame after removing missing values:", len(new_data))
    print("Number of rows with at least 1 NA value: ", (len(nrc_profile_df) - len(new_data)))

# as the baseline,prepare Mean and SD from Big Five personality scores of all users
    print(nrc_profile_df.mean())
    print(nrc_profile_df.std())
    female = np.sum(nrc_profile_df['gender'] > 0.5)
    print('Female users: ', female)
    print('Male users: ', 9500 - female)
    # Create a radar chart for personality traits
    personality_df = nrc_profile_df.iloc[:,-5:]
    gender_df = nrc_profile_df.iloc[:,-6]
    age_df = nrc_profile_df.iloc[:,-7]
    # Histogram
    NRC_Classifier.plot_personality_histograms(personality_df)
    labels = ['Openness', 'Conscientiousness', 'Extroversion', 'Agreeableness', 'Neuroticism']
    values = personality_df.mean()
    title = 'Personality Traits Mean Radar Chart (0.0 - 5.0)'
    NRC_Classifier.radar_chart(labels, values, title)
    # Create a radar chart for emotion features
    emotion_df = nrc_profile_df.iloc[:,1:11]
    #Min-Max scaling
    X=(emotion_df-emotion_df.min())/(emotion_df.max()-emotion_df.min())
    X.hist()
    plt.show()
    labels = list(emotion_df)[:]
    values = emotion_df.mean()
    title = 'Emotion Features Mean Radar Chart (0.0 - 1.0)'
    NRC_Classifier.radar_chart(labels, values, title)

# prepare emotion features correlation for each of the personality traits
    df_personality_emotion = pd.concat([personality_df, emotion_df], axis=1)
    correlation = df_personality_emotion.corr(method='pearson', min_periods=1)
    correlation.to_csv("../resources/personality_emotion_correlation.csv")

# prepare accuracy result table for personality traits
    for i in range(5):
        NRC_Classifier.variable_predictor_linreg(emotion_df, personality_df.iloc[:,i], personality_df.columns[i])



