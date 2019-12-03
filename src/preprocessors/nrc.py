"""
 Ran on 24, Nov : [0.61674547 0.69935273 0.80564432 0.65389657 0.7969793 ]
(all)liwc, nrc,image:[0.62744298 0.70276885 0.79951852 0.64674137 0.78883464]

                                        'ope',      'con',      'ext',     'agr',     'neu'
  liwc nrc       (Linear regression) : [0.62920075 0.71032693 0.81649614 0.66209088 0.79016244]
l_m.RidgeCV(alphas=np.logspace(-6, 6, 13))[0.62696862 0.70135399 0.80954163 0.6535788  0.78303217]
  liwc nrc       (RidgeCV) :           [0.62873563 0.71021455 0.8163051  0.66174107 0.78974958]
  liwc,nrc,pix   (Linear regression) : [0.63021307 0.70558353 0.8146352  0.65851407 0.78832862]
   liwc,nrc,pix   (RidgeCV)          : [0.62974593 0.70506652 0.81416499 0.65802158 0.78796138]
   liwc,nrc,pix   (Lasso)            : [0.6318923  0.71527513 0.80450027 0.65765008 0.79144684]
   liwc,nrc,pix   (Ridge)            : [0.63012872 0.70540048 0.81450497 0.65832511 0.78826474]
 liwc,nrc,pix(linear_model.LassoLars ):[0.63230685 0.71530043 0.80542392 0.65807075 0.79141732]
 L,n,plinear_model.MultiTaskElasticNet [0.63121542 0.71447838 0.80586788 0.65591106 0.7905716 ]
 L,n,plinear_model.ElasticNet          [0.63085372 0.71493107 0.80506929 0.65576473 0.7907966 ]
 L,n,plinear_model.MultiTaskLasso      [0.63023951 0.71475692 0.80398694 0.65673558 0.79096756]

 ext:  (December 1 results)
             MultiTaskElasticNet,    0.8050692930709286
             MultiTaskLasso,         0.8045002727566398  ---- Least Error
             ElasticNet,             0.808223816014654
             LassoLars,              0.8054239165007422
             Ridge:                  0.8145049659596015
.RidgeCV(alphas=np.logspace(-6, 6, 13), 0.8138380953062951
LinearRegression                       0.814635202620293
LinearRegression(liwc,nrc)             0.8164961415169353
LinearRegression(liwc,nrc,image)      0.814635202620293

neu:
MultiTaskLasso:                            0.7914468413341056
RidgeCV(np.logspace(-6, 6, 13)):           0.7830321701603294  --- Least Error
LinearRegression                           0.7883286209831646
MultiTaskElasticNet                        0.7907966032516445
MultiTaskLasso                             0.7914468413341056
 """
import numpy as np
import pandas as pd
from sklearn import linear_model

from util import Utils

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_data(labels=['userid', 'ope', 'con', 'ext', 'agr', 'neu']):
    util = Utils()
    profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
    profile_df = profile_df.filter(labels, axis=1)

    nrc_df = util.read_data_to_dataframe("../../data/Train/Text/nrc.csv")
    liwc_df = util.read_data_to_dataframe("../../data/Train/Text/liwc.csv")
    nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
    liwc_df.rename(columns={'userId': 'userid'}, inplace=True)
    image_df = read_image()
    merged_df = pd.merge(nrc_df, liwc_df, on='userid')
    merged_df = pd.merge(merged_df, image_df, on='userid')
    merged_df = pd.merge(merged_df, profile_df, on='userid')
    merged_df.drop(['userid'], axis=1, inplace=True)
    return merged_df


def read_image(profiles_path="../../data/Train/Profile/Profile.csv",
               image_path="../../data/Train/Image/oxford.csv"):
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


if __name__ == '__main__':
    """
    Ran on Dec,2 :
    ext:
    liwc,nrc,image (RidgeCV(np.logspace(-6, 6, 13)))  0.7830321701603294  ,alpha=10000
    ext:
    liwc,nrc,image (LinearRegression)                 0.814635202620293  
    """

    df = get_data(labels=['userid', 'neu'])
    X = df.iloc[:, 0:-1]  # independent columns
    X = np.log(X + 1)
    X = (X - X.min()) / (X.max() - X.min())
    X.fillna(0, inplace=True)
    util = Utils()

    y = df[df.columns[-1:]]
    util = Utils()
    reg = linear_model.RidgeCV(np.logspace(-6, 6, 13))


    # reg.fit(X, y)
    util.perform_cross_validation(reg,df)
    # pickle.dump(reg, open("../resources/LinearRegression_ext.sav", 'wb'))
