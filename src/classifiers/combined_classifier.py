import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings


from util import Utils

warnings.filterwarnings("ignore")


class CombinedClassifier:

    @staticmethod
    def read_liwc(profiles_path="../../data/Train/Profile/Profile.csv",
                  liwc_path="../../data/Train/Text/liwc.csv"):
        util = Utils()
        profile_df = util.read_data_to_dataframe(profiles_path)
        liwc_df = util.read_data_to_dataframe(liwc_path)
        liwc_df.rename(columns={'userId': 'userid'}, inplace=True)
        merged_df = pd.merge(profile_df, liwc_df, on='userid', how='left')

        return merged_df.filter(
            ['userid', 'WC', 'WPS', 'Sixltr', 'Dic', 'Numerals', 'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe',
             'they', 'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb', 'preps',
             'conj', 'negate', 'quant', 'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect',
             'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat',
             'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health',
             'sexual', 'ingest', 'relativ', 'motion', 'space', 'time', 'work', 'achieve', 'leisure', 'home',
             'money', 'relig', 'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon', 'SemiC',
             'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP', 'AllPct', 'age'], axis=1)

    @staticmethod
    def read_image(profiles_path="../../data/Train/Profile/Profile.csv",
                   image_path="../../data/Train/Image/oxford.csv", is_train=True):
        util = Utils()
        profile_df = util.read_data_to_dataframe(profiles_path)
        profile_df.drop(profile_df.columns.difference(['userid', 'age']), 1, inplace=True)
        image_df = util.read_data_to_dataframe(image_path)
        image_df.rename(columns={'userId': 'userid'}, inplace=True)
        merged_df = pd.merge(image_df, profile_df, on='userid')
        merged_df = merged_df.filter(
            ['userid', 'faceRectangle_width', 'faceRectangle_height', 'faceRectangle_left', 'faceRectangle_top',
             'pupilLeft_x', 'pupilLeft_y', 'pupilRight_x', 'pupilRight_y', 'noseTip_x', 'noseTip_y', 'mouthLeft_x',
             'mouthLeft_y', 'mouthRight_x', 'mouthRight_y', 'eyebrowLeftOuter_x', 'eyebrowLeftOuter_y',
             'eyebrowLeftInner_x', 'eyebrowLeftInner_y', 'eyeLeftOuter_x', 'eyeLeftOuter_y', 'eyeLeftTop_x',
             'eyeLeftTop_y', 'eyeLeftBottom_x', 'eyeLeftBottom_y', 'eyeLeftInner_x', 'eyeLeftInner_y',
             'eyebrowRightInner_x', 'eyebrowRightInner_y', 'eyebrowRightOuter_x', 'eyebrowRightOuter_y',
             'eyeRightInner_x', 'eyeRightInner_y', 'eyeRightTop_x', 'eyeRightTop_y', 'eyeRightBottom_x',
             'eyeRightBottom_y', 'eyeRightOuter_x', 'eyeRightOuter_y', 'noseRootLeft_x', 'noseRootLeft_y',
             'noseRootRight_x', 'noseRootRight_y', 'noseLeftAlarTop_x', 'noseLeftAlarTop_y', 'noseRightAlarTop_x',
             'noseRightAlarTop_y', 'noseLeftAlarOutTip_x', 'noseLeftAlarOutTip_y', 'noseRightAlarOutTip_x',
             'noseRightAlarOutTip_y', 'upperLipTop_x', 'upperLipTop_y', 'upperLipBottom_x', 'upperLipBottom_y',
             'underLipTop_x', 'underLipTop_y', 'underLipBottom_x', 'underLipBottom_y', 'facialHair_mustache',
             'facialHair_beard', 'facialHair_sideburns', 'headPose_roll', 'headPose_yaw', 'headPose_pitch'
                , 'age'], axis=1)
        if is_train:
            merged_df['age'] = pd.cut(merged_df['age'], [0, 25, 35, 50, 200],
                                      labels=["xx-24", "25-34", "35-49", "50-xx"],
                                      right=False)
        return merged_df

    def merge_images_piwc(self, is_train=True, profiles_path="../../data/Train/Profile/Profile.csv",
                          liwc_path="../../data/Train/Text/liwc.csv",
                          image_path="../../data/Train/Image/oxford.csv"):
        df_liwc = self.read_liwc(profiles_path, liwc_path)
        if is_train:
            df_liwc.drop(['age'], axis=1, inplace=True)
        image_df = self.read_image(profiles_path, image_path, is_train)
        merged_df = pd.merge(image_df, df_liwc, on='userid', how='left')
        merged_df.drop(['userid'], axis=1, inplace=True)
        if is_train:
            merged_df["ages"] = merged_df["age"]
            merged_df.drop(['age'], axis=1, inplace=True)
        return merged_df

    @staticmethod
    def logistic_regression_customized(df):
        X_train, X_test, y_train, y_test = Utils.split_data(df)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_train)
        y_df = pd.DataFrame(y_pred)
        chos = y_df.max(axis=1)
        chos_df = pd.DataFrame(chos)
        indexes = set()
        for index, row in chos_df.iterrows():
            if row[0] < 0.30:
                indexes.add(index)

        y_prediction = clf.predict(X_test)

        for j in range(y_prediction.shape[0]):
            if j in indexes:
                y_prediction[j] = "xx-24"

        print("logistic regression acc: ", accuracy_score(y_test, y_prediction))

    @staticmethod
    def run_classifier_for_accuracy(df, clf):
        data = df.to_numpy()
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        accuracy = cross_val_score(clf, X, y, cv=10)
        print("Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%")

    @staticmethod
    def predict_age_using_logistic_regression(df):
        df_age=df["ages"]
        df =df.filter(["faceRectangle_left", "faceRectangle_top", "pupilLeft_x", "pupilLeft_y", "pupilRight_x", "pupilRight_y", "noseTip_x", "noseTip_y", "mouthLeft_x", "mouthLeft_y", "mouthRight_x", "mouthRight_y", "eyebrowLeftOuter_x", "eyebrowLeftOuter_y", "eyebrowLeftInner_x", "eyebrowLeftInner_y", "eyeLeftOuter_x", "eyeLeftOuter_y", "eyeLeftTop_y", "eyeLeftBottom_x", "eyeLeftBottom_y", "eyeLeftInner_x", "eyeLeftInner_y", "eyebrowRightInner_x", "eyebrowRightInner_y", "eyebrowRightOuter_x", "eyebrowRightOuter_y", "eyeRightInner_x", "eyeRightInner_y", "eyeRightTop_x", "eyeRightTop_y", "eyeRightBottom_x", "eyeRightBottom_y", "eyeRightOuter_x", "eyeRightOuter_y", "noseRootLeft_x", "noseRootLeft_y", "noseRootRight_y", "noseLeftAlarTop_x", "noseLeftAlarTop_y", "noseRightAlarTop_x", "noseRightAlarTop_y", "noseLeftAlarOutTip_x", "noseLeftAlarOutTip_y", "noseRightAlarOutTip_x", "noseRightAlarOutTip_y", "upperLipTop_x", "upperLipTop_y", "upperLipBottom_x", "upperLipBottom_y", "underLipTop_x", "underLipTop_y", "underLipBottom_x", "underLipBottom_y", "facialHair_mustache", "facialHair_beard", "facialHair_sideburns", "Sixltr", "Dic", "Numerals", "funct", "pronoun", "ppron", "i", "we", "shehe", "they", "article", "verb", "auxverb", "past", "present", "future", "adverb", "preps", "conj", "negate", "quant", "number", "swear", "social", "family", "friend", "humans", "affect", "posemo", "negemo", "anx", "anger", "sad", "cogmech", "insight", "cause", "discrep", "tentat", "certain", "inhib", "incl", "excl", "percept", "see", "hear", "feel", "bio", "body", "health", "sexual", "ingest", "work", "achieve", "leisure", "home", "money", "relig", "death", "assent", "nonfl", "filler", "Period", "Comma", "Colon", "SemiC", "QMark", "Exclam", "Dash", "Quote", "Apostro", "Parenth", "OtherP", "AllPct"],axis=1)
        df["age"]=df_age
        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1]
        clf = LogisticRegression(C=0.004832930238571752,
                                 penalty='l2')
        clf.fit(X, y)
        pickle.dump(clf, open("../resources/LogisticRegressionAge_v2.sav", 'wb'))

    def fit_model_using_default_ica_rfe(self, clf, df):
        # print("model with all features")
        # self.run_classifier_for_accuracy(df, clf)
        # df_ica = UTILS.apply_fast_ica(df, 10,1)
        # print("model with 10 features ica applied")
        # self.run_classifier_for_accuracy(df_ica, clf)
        # for i in range(124,129):

        df_rfe = UTILS.apply_rfe(df, clf, 130,1)
        print("model with 10 features rfe applied")
        self.run_classifier_for_accuracy(df_rfe, clf)


if __name__ == "__main__":
    UTILS = Utils()
    clf = LogisticRegression(C=0.004832930238571752,
                             penalty='l2')
    CC = CombinedClassifier()
    df = CC.merge_images_piwc()
    # CC.fit_model_using_default_ica_rfe(clf, df)
    CC.predict_age_using_logistic_regression(df)
