import pickle

from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from src.util import Utils
import pandas as pd
from sklearn.metrics import accuracy_score, hinge_loss
import numpy as np
import matplotlib.pyplot as plt

class ImageClassifier:
    @staticmethod
    def get_image_gender_training_data():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        profile_df.drop(profile_df.columns.difference(['userid', 'gender']), 1, inplace=True)
        image_df = util.read_data_to_dataframe("../../data/Train/Image/oxford.csv")
        image_df.rename(columns={'userId': 'userid'}, inplace=True)
        merged_df = pd.merge(image_df, profile_df, on='userid')
        return merged_df.filter(
            ['faceRectangle_width', 'faceRectangle_height', 'faceRectangle_left', 'faceRectangle_top',
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
                , 'gender'], axis=1)

    @staticmethod
    def get_image_age_training_data():
        util = Utils()
        profile_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        profile_df.drop(profile_df.columns.difference(['userid', 'age']), 1, inplace=True)
        image_df = util.read_data_to_dataframe("../../data/Train/Image/oxford.csv")
        image_df.rename(columns={'userId': 'userid'}, inplace=True)
        merged_df = pd.merge(image_df, profile_df, on='userid')
        merged_df =  merged_df.filter(
            ['faceRectangle_width', 'faceRectangle_height', 'faceRectangle_left', 'faceRectangle_top',
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

    @staticmethod
    def sgd_classify(df_gender):
        X_train, X_test, y_train, y_test = Utils.split_data(df_gender)
        clf = SGDClassifier(loss="hinge", penalty="l2")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("sgd acc: ", accuracy_score(y_test, y_pred))

    @staticmethod
    def knn_classify(df_gender, predicted_variable):
        X_train, X_test, y_train, y_test = Utils.split_data(df_gender)
        neigh = KNeighborsClassifier(n_neighbors=18)
        neigh.fit(X_train, y_train)
        pickle.dump(neigh, open("../resources/KNNimages_" + predicted_variable + ".sav", 'wb'))
        y_pred = neigh.predict(X_test)
        print("KNN acc: ", accuracy_score(y_test, y_pred))

    @staticmethod
    def kernel_estimation(df_gender):
        X_train, X_test, y_train, y_test = Utils.split_data(df_gender)
        rbf_feature = RBFSampler()
        X_features = rbf_feature.fit_transform(X_train)
        clf = SGDClassifier()
        clf.fit(X_features,y_train)
        print("Kernel Density acc: ", clf.score(X_features, y_train))

    @staticmethod
    def svm_estimation(df_gender):
        X_train, X_test, y_train, y_test = Utils.split_data(df_gender)
        clf = svm.SVC(gamma='scale')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("SVM acc: ", accuracy_score(y_test, y_pred))

    @staticmethod
    def random_forest_classifier(df_gender):
        X_train = df_gender.iloc[:, :-1]
        y_train = df_gender.iloc[:, -1]
        clf = RandomForestClassifier(bootstrap=False, n_estimators=400, max_depth=10, max_features='sqrt')
        clf.fit(X_train, y_train)
        pickle.dump(clf, open("../resources/RandomForest_Gender.sav", 'wb'))

    @staticmethod
    def random_forest_classifier_kfold_validation(df_gender):
        data = df_gender.to_numpy()
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestClassifier(n_estimators=10)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("Random Forest acc: ", accuracy_score(y_test, y_pred))

    @staticmethod
    def plot_gender_histograms(df_gender):
        data = df_gender.to_numpy()
        x = data[:,:-1]
        y = data[:,-1]
        standard_scaler = preprocessing.StandardScaler()
        x_scaled = standard_scaler.fit_transform(x)
        df = pd.concat([pd.DataFrame(x_scaled), pd.DataFrame(y)], axis=1)

        females = df.loc[df.iloc[:, -1] == 1]
        males = df.loc[df.iloc[:, -1] == 0]
        for i in range(females.shape[1] - 1):
            plt.hist([females.iloc[:, i], males.iloc[:, i]], color=['orange', 'green'], density=True)
            plt.legend(['female', 'male'])
            plt.ylabel(list(df_gender)[i])
            plt.show()

    @staticmethod
    def plot_age_histograms(df_age):
        data = df_age.to_numpy()
        x = data[:,:-1]
        y = data[:,-1]
        standard_scaler = preprocessing.StandardScaler()
        x_scaled = standard_scaler.fit_transform(x)
        df = pd.concat([pd.DataFrame(x_scaled), pd.DataFrame(y)], axis=1)

        x0_24 = df.loc[df.iloc[:, -1] == "xx-24"]
        x25_34 = df.loc[df.iloc[:, -1] == "25-34"]
        x35_49 = df.loc[df.iloc[:, -1] == "35-49"]
        x50_xx = df.loc[df.iloc[:, -1] == "50-xx"]
        for i in range(x0_24.shape[1] - 1):
            plt.hist([x0_24.iloc[:, i], x25_34.iloc[:, i], x35_49.iloc[:, i], x50_xx.iloc[:, i]], color=['orange', 'lime', 'red', 'blue'], density=True)
            plt.legend(['0_24', '25_34', '35_49', '50_xx'])
            plt.ylabel(list(df_age)[i])
            plt.show()

if __name__ == "__main__":
    IMAGE_CLASSIFIER = ImageClassifier()
    df_gender = IMAGE_CLASSIFIER.get_image_gender_training_data()
    # IMAGE_CLASSIFIER.random_forest_classifier_kfold_validation(df_gender)
    # IMAGE_CLASSIFIER.sgd_classify(df_gender)
    # IMAGE_CLASSIFIER.knn_classify(df_gender)
    # IMAGE_CLASSIFIER.kernel_estimation(df_gender)
    # IMAGE_CLASSIFIER.svm_estimation(df_gender)
    IMAGE_CLASSIFIER.random_forest_classifier(df_gender)
    df_age_group = IMAGE_CLASSIFIER.get_image_age_training_data()
    # IMAGE_CLASSIFIER.sgd_classify(df_age_group)
    # IMAGE_CLASSIFIER.knn_classify(df_age_group, "age-group")
    # IMAGE_CLASSIFIER.kernel_estimation(df_age_group)
    # IMAGE_CLASSIFIER.svm_estimation(df_age_group)
    # IMAGE_CLASSIFIER.random_forest_classifier(df_age_group)
    # IMAGE_CLASSIFIER.plot_gender_histograms(df_gender)
    # IMAGE_CLASSIFIER.plot_age_histograms(df_age_group)
    # gavazn! = ""
