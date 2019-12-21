import pickle

import numpy as np
import pandas as pd
from sklearn import linear_model

from util import Utils

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Personality:

    def get_data(self, labels=['userid', 'ope', 'con', 'ext', 'agr', 'neu'], include_image=False):
        util = Utils()
        profile_df = util.read_data_to_dataframe("../data/Train/Profile/Profile.csv")
        profile_df = profile_df.filter(labels, axis=1)

        nrc_df = util.read_data_to_dataframe("../data/Train/Text/nrc.csv")
        liwc_df = util.read_data_to_dataframe("../data/Train/Text/liwc.csv")
        nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
        liwc_df.rename(columns={'userId': 'userid'}, inplace=True)
        if include_image:
            image_df = self.read_image()
        merged_df = pd.merge(nrc_df, liwc_df, on='userid')
        if include_image:
            merged_df = pd.merge(merged_df, image_df, on='userid')
        merged_df = pd.merge(merged_df, profile_df, on='userid')
        merged_df.drop(['userid'], axis=1, inplace=True)
        return merged_df

    @staticmethod
    def read_image(profiles_path="../data/Train/Profile/Profile.csv",
                   image_path="../data/Train/Image/oxford.csv"):
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
             'facialHair_beard', 'facialHair_sideburns', 'headPose_roll', 'headPose_yaw', 'headPose_pitch'], axis=1)
        return merged_df

    @staticmethod
    def generate_ext_model():
        PERSONALITY = Personality()
        df = PERSONALITY.get_data(labels=['userid', 'ext'])
        df = df.filter(
            ['positive', 'negative', 'anger_x', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise',
             'trust',
             'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'future', 'affect', 'posemo', 'negemo',
             'anx',
             'incl', 'work', 'death', 'assent', 'nonfl', 'Quote', 'Apostro', 'ext'], axis=1)
        reg = linear_model.LinearRegression()

        X = Personality.normalize(df)
        y = df[df.columns[-1:]]
        reg.fit(X, y)
        pickle.dump(reg, open("../resources/LinearRegression_ext_v2.sav", 'wb'))

    @staticmethod
    def generate_all_personality_model():
        PERSONALITY = Personality()
        df = PERSONALITY.get_data(labels=['userid', 'ope', 'con', 'ext', 'agr', 'neu'], include_image=False)
        reg = linear_model.LinearRegression()
        X = Personality.normalize(df)
        X.drop(['ope', 'con', 'ext', 'agr'], axis=1, inplace=True)
        y = df[df.columns[-5:]]
        reg.fit(X, y)
        pickle.dump(reg, open("resources/LinearReg_Personality.sav", 'wb'))

    @staticmethod
    def generate_neu_model():
        PERSONALITY = Personality()
        df = PERSONALITY.get_data(labels=['userid', 'neu'], include_image=True)
        df = df.filter(
            ['positive', 'negative', 'anger_x', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise',
             'trust',
             'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'future', 'affect', 'posemo', 'negemo',
             'anx',
             'incl', 'work', 'death', 'assent', 'nonfl', 'Quote', 'Apostro', 'ext'], axis=1)
        reg = linear_model.RidgeCV()
        X = Personality.normalize(df)
        y = df[df.columns[-1:]]
        reg.fit(X, y)
        pickle.dump(reg, open("resources/RidgeCV_neu.sav", 'wb'))

    @staticmethod
    def generate_ext_model():
        PERSONALITY = Personality()
        df = PERSONALITY.get_data(labels=['userid', 'ext'])
        df = df.filter(
            ['positive', 'negative', 'anger_x', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise',
             'trust',
             'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'future', 'affect', 'posemo', 'negemo',
             'anx',
             'incl', 'work', 'death', 'assent', 'nonfl', 'Quote', 'Apostro', 'ext'], axis=1)
        reg = linear_model.LinearRegression()

        X = Personality.normalize(df)
        y = df[df.columns[-1:]]
        reg.fit(X, y)
        pickle.dump(reg, open("resources/LinearRegression_ext_v2.sav", 'wb'))

    @staticmethod
    def normalize(df):
        X = df.iloc[:, 0:-1]  # independent columns
        X = np.log(X + 1)
        X = (X - X.min()) / (X.max() - X.min())
        X.fillna(0, inplace=True)
        return X


if __name__ == '__main__':
    util = Utils()
    PERSONALITY = Personality()
    df = PERSONALITY.get_data(labels=['userid', 'ext'])
    df = df.filter(
        ['positive', 'negative', 'anger_x', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust',
         'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'future', 'affect', 'posemo', 'negemo', 'anx',
         'incl', 'work', 'death', 'assent', 'nonfl', 'Quote', 'Apostro', 'ext'], axis=1)
    reg = linear_model.LinearRegression()

    X = df.iloc[:, 0:-1]  # independent columns
    X = np.log(X + 1)
    X = (X - X.min()) / (X.max() - X.min())
    X.fillna(0, inplace=True)

    y = df[df.columns[-1:]]

    reg.fit(X, y)
    # util.perform_cross_validation(reg, df)
    pickle.dump(reg, open("resources/LinearRegression_ext_v2.sav", 'wb'))
