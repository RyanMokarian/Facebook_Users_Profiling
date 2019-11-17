import numpy as np
import pandas as pd
from nltk.classify import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.util import Utils


class CombinedClassifier():

    @staticmethod
    def read_liwc():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        liwc_df = util.read_data_to_dataframe("../../data/Train/Text/liwc.csv")
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
    def read_image():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        profile_df.drop(profile_df.columns.difference(['userid', 'age']), 1, inplace=True)
        image_df = util.read_data_to_dataframe("../../data/Train/Image/oxford.csv")
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
        merged_df['age'] = pd.cut(merged_df['age'], [0, 25, 35, 50, 200], labels=["xx-24", "25-34", "35-49", "50-xx"],
                                  right=False)
        return merged_df

    def merge_images_piwc(self):
        df_liwc = self.read_liwc()
        image_df = self.read_image()
        merged_df = pd.merge(image_df, df_liwc, on='userid', how='left')
        merged_df.drop(['userid'], axis=1, inplace=True)
        merged_df["ages"] = merged_df["age"]
        merged_df.drop(['age'], axis=1, inplace=True)
        return merged_df

    @staticmethod
    def logistic_regression(df):
        """
        All features :            Accuracy: 61.91%
        RFE 10 selected features :Accuracy: 59.71%
        :param df:
        :return:
        """
        data = df.to_numpy()
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        clf = LogisticRegression()
        accuracy = cross_val_score(clf, X, y, cv=10)
        print("Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%")

    def random_forest_classifier_kfold_validation(df):
        data = df.to_numpy()
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        clf = RandomForestClassifier(n_estimators=10)
        accuracy = cross_val_score(clf, X, y, cv=10)
        print("Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%")

    def svm_estimation(df):
        """
        All features             :Accuracy: 61.92%
        RFE 10 selected features :Accuracy: 59.67%
        :param df:
        :return:
        """
        data = df.to_numpy()
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        clf = svm.LinearSVC()
        accuracy = cross_val_score(clf, X, y, cv=10)
        print("Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%")
