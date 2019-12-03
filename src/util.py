import csv
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")


class Utils:

    def read_json(self, filename):
        """
        This method loads a json from a file
        :param filename:
        :return: a json file
        """
        resources = {}
        if filename:
            with open(filename, 'r') as file:
                resources = json.load(file)
        return resources

    def write_json_to_directory(self, json_object, file_name, directory="resources"):
        """
        This method writes a python object to a json file
        """
        self.make_directory_if_not_exists(directory)
        filename = os.path.join(directory, file_name)
        with open(filename, 'w') as outfile:
            json.dump(json_object, outfile)

    @staticmethod
    def write_to_directory(path, content):
        """
        This method writes a python object to a json file
        """
        with open(path, "w") as text_file:
            print(content, file=text_file)

    @staticmethod
    def make_directory_if_not_exists(directory):
        """
        this method creates a directory if it does not exists
        :param directory:
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def read_csv(filename):
        """
        This method reads a CSV file to a list
        :param filename:
        :return: a list containing elements per line of CSV
        """
        with open(filename, 'r') as a_file:
            return list(csv.reader(a_file))

    @staticmethod
    def read_data_to_dataframe(path):
        print(os.path.abspath(__file__))
        return pd.read_csv(path)

    @staticmethod
    def split_data(df):
        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def split_data_n_columns(df, n_columns):
        data = df.to_numpy()
        X = data[:, :n_columns]
        y = data[:, n_columns]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def perform_cross_validation(clf, df, n_columns=1):
        """
        This method is used to do  the cross validation
        :param clf:
        :param df:
        :return:
        """

        kf = KFold(n_splits=10)
        arr = []
        for train_indices, test_indices in kf.split(df):
            clf.fit(df.iloc[train_indices, :-n_columns], df.iloc[train_indices, -n_columns:])
            print(clf.alpha_)
            pred = clf.predict(df.iloc[test_indices, :-n_columns])
            y_yest = df.iloc[test_indices, -n_columns:].to_numpy()
            if n_columns == 1:
                arr.append(np.average(np.sqrt(np.average((y_yest - pred) ** 2))))
            else:
                arr.append(np.sqrt(np.average((y_yest - pred) ** 2, axis=0)))
        print("cross validation: " + str(np.average(arr)))

    @staticmethod
    def read_pickle_from_file(file_name):
        with open(file_name, 'rb') as input:
            return pickle.load(input)

    @staticmethod
    def write_object_to_file(obj, file_name):
        with open(file_name, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def apply_rfe(df, clf, n_features_to_select, n_labels):
        """
        This method applied recursive feature elimination on a model to
        select n_features_to_select of the most important features
        :param df: dataframe [X,y]
        :param clf: the model
        :param n_features_to_select:
        :return: [X,y]
        """
        labels_name = list(df.columns.values)[len(list(df.columns.values)) - 1]
        df_y = df.iloc[:, -1]
        data = df.to_numpy()
        np.random.shuffle(data)
        X = data[:, :-n_labels]
        y = data[:, -n_labels]
        selector = RFE(clf, n_features_to_select, step=1, verbose=1)
        selector = selector.fit(X, y)
        print("Important  features:" + str(df.columns[np.where(selector.ranking_ == 1)[0]]))
        df = df[df.columns[np.where(selector.ranking_ == 1)[0]]]
        df[labels_name] = df_y
        return df

    @staticmethod
    def apply_fast_ica(df, number_of_components, number_of_lables):
        """
        This method applied ICA to a data_frame providing the number of components
        :param df:
        :param number_of_components:
        :return:
        """
        transformer = FastICA(n_components=number_of_components)
        labels_name = list(df.columns.values)[len(list(df.columns.values)) - 1]
        df_y = df.iloc[:, -number_of_lables]
        df = df.iloc[:, :-number_of_lables]
        data_transformed = transformer.fit_transform(df)
        data_transformed_df = pd.DataFrame(data_transformed)
        data_transformed_df[labels_name] = df_y
        return data_transformed_df

    @staticmethod
    def remove_rows_with_condition(df, condition):
        """
        this method removes rows from data frame based on some conditions
        :param df:
        :param condition:
        :return:
        """
        df = df[condition]
        return df

    @staticmethod
    def normalize_df(df):
        """
        this method normalizes a data_frame
        :param df:
        :return:
        """
        labels_name = list(df.columns.values)[len(list(df.columns.values)) - 1]
        df_y = df.iloc[:, -1]
        df = df.iloc[:, :-1]
        normalized_df = (df - df.min()) / (df.max() - df.min())
        normalized_df[labels_name] = df_y
        return normalized_df

    @staticmethod
    def run_exhaustive_search(clf, df, columns, parameter_space):
        """
        used the code from:
        https://datascience.stackexchange.com/questions/36049/
        how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
        :param clf:
        :param df:
        :param parameter_space:
        :return:
        """
        X_train, X_test, y_train, y_test = Utils.split_data_n_columns(df, columns)

        clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
        clf.fit(X_train, y_train)

        print('Best parameters found:\n', clf.best_params_)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        y_true, y_pred = y_test, clf.predict(X_test)

        from sklearn.metrics import classification_report
        print('Results on the test set:')
        print(classification_report(y_true, y_pred))
