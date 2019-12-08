import os
import pickle
import pandas as pd
import numpy as np

from classifiers.combined_classifier import CombinedClassifier
from preprocessors.nrc import read_image
from util import Utils

abs_path = os.path.dirname(os.path.abspath(__file__))


def compute_predictions_for_age_group(X_test):
    model_path = os.path.join(abs_path, os.path.join("resources", "KNNlikes_age-group.sav"))
    loaded_model = pickle.load(open(model_path, 'rb'))
    y_pred = loaded_model.predict(X_test)
    pass


def compute_gender(test_data_path, df_results):
    model_path = os.path.join(abs_path, os.path.join("resources", "RandomForest_Gender.sav"))
    profile_df = Utils.read_data_to_dataframe(test_data_path + "/Profile/Profile.csv")
    profile_df.drop(profile_df.columns.difference(['userid', 'gender']), 1, inplace=True)
    image_df = Utils.read_data_to_dataframe(test_data_path + "/Image/oxford.csv")
    image_df.rename(columns={'userId': 'userid'}, inplace=True)

    merged_df = pd.merge(image_df, profile_df, on='userid')
    merged_df.drop(['userid', 'faceID', 'gender'], axis=1, inplace=True)
    model = Utils.read_pickle_from_file(model_path)

    model.predict(merged_df)
    image_df["gender"] = model.predict(merged_df)
    predicted_df = profile_df["userid"].to_frame()

    # image_df['userid'] = image_df['userid'].astype('|S')
    predicted_df = pd.merge(predicted_df, image_df, on="userid", how="left")
    user_gender_df = predicted_df.filter(["userid", "gender"])
    user_gender_df["gender"].fillna(1, inplace=True)
    user_gender_df = aggregate_duplicate_ids(user_gender_df, 'gender')

    df_results = pd.merge(df_results, user_gender_df, on='userid', how="left")
    df_results.drop(['gender_x'], axis=1, inplace=True)
    df_results.rename(columns={"gender_y": "gender"}, inplace=True)

    df_results.loc[df_results.gender == 0, 'gender'] = "male"
    df_results.loc[df_results.gender == 1, 'gender'] = "female"
    return df_results


def generate_df_for_all_users(profiles, model):
    profiles["age"] = model['age_group']
    profiles["gender"] = model['gender']
    profiles["ope"] = model['open']
    profiles["con"] = model['conscientious']
    profiles["ext"] = model['extrovert']
    profiles["agr"] = model['agreeable']
    profiles["neu"] = model['neurotic']
    return profiles


def compute_age(test_data_path, df_results):
    model_path = os.path.join(abs_path, os.path.join("resources", "LogisticRegressionAge_v2.sav"))
    profile_df = Utils.read_data_to_dataframe(test_data_path + "/Profile/Profile.csv")

    profile_df.drop(profile_df.columns.difference(['userid', 'age']), 1, inplace=True)
    image_df = Utils.read_data_to_dataframe(test_data_path + "/Image/oxford.csv")
    image_df.rename(columns={'userId': 'userid'}, inplace=True)

    combined_classifier = CombinedClassifier()
    merged_df = combined_classifier.merge_images_piwc(is_train=False,
                                                      profiles_path=test_data_path + "/Profile/Profile.csv",
                                                      liwc_path=test_data_path + "/Text/liwc.csv",
                                                      image_path=test_data_path + "/Image/oxford.csv")

    merged_df.drop(['age_x', 'age_y'], axis=1, inplace=True)
    merged_df = merged_df.filter(
        ["faceRectangle_left", "faceRectangle_top", "pupilLeft_x", "pupilLeft_y", "pupilRight_x", "pupilRight_y",
         "noseTip_x", "noseTip_y", "mouthLeft_x", "mouthLeft_y", "mouthRight_x", "mouthRight_y", "eyebrowLeftOuter_x",
         "eyebrowLeftOuter_y", "eyebrowLeftInner_x", "eyebrowLeftInner_y", "eyeLeftOuter_x", "eyeLeftOuter_y",
         "eyeLeftTop_y", "eyeLeftBottom_x", "eyeLeftBottom_y", "eyeLeftInner_x", "eyeLeftInner_y",
         "eyebrowRightInner_x", "eyebrowRightInner_y", "eyebrowRightOuter_x", "eyebrowRightOuter_y", "eyeRightInner_x",
         "eyeRightInner_y", "eyeRightTop_x", "eyeRightTop_y", "eyeRightBottom_x", "eyeRightBottom_y", "eyeRightOuter_x",
         "eyeRightOuter_y", "noseRootLeft_x", "noseRootLeft_y", "noseRootRight_y", "noseLeftAlarTop_x",
         "noseLeftAlarTop_y", "noseRightAlarTop_x", "noseRightAlarTop_y", "noseLeftAlarOutTip_x",
         "noseLeftAlarOutTip_y", "noseRightAlarOutTip_x", "noseRightAlarOutTip_y", "upperLipTop_x", "upperLipTop_y",
         "upperLipBottom_x", "upperLipBottom_y", "underLipTop_x", "underLipTop_y", "underLipBottom_x",
         "underLipBottom_y", "facialHair_mustache", "facialHair_beard", "facialHair_sideburns", "Sixltr", "Dic",
         "Numerals", "funct", "pronoun", "ppron", "i", "we", "shehe", "they", "article", "verb", "auxverb", "past",
         "present", "future", "adverb", "preps", "conj", "negate", "quant", "number", "swear", "social", "family",
         "friend", "humans", "affect", "posemo", "negemo", "anx", "anger", "sad", "cogmech", "insight", "cause",
         "discrep", "tentat", "certain", "inhib", "incl", "excl", "percept", "see", "hear", "feel", "bio", "body",
         "health", "sexual", "ingest", "work", "achieve", "leisure", "home", "money", "relig", "death", "assent",
         "nonfl", "filler", "Period", "Comma", "Colon", "SemiC", "QMark", "Exclam", "Dash", "Quote", "Apostro",
         "Parenth", "OtherP", "AllPct"], axis=1)

    model = Utils.read_pickle_from_file(model_path)
    image_df["age_group"] = model.predict(merged_df)

    predicted_df = profile_df["userid"].to_frame()
    predicted_df = pd.merge(predicted_df, image_df, on='userid', how="left")
    user_age_df = predicted_df.filter(["userid", "age_group"])
    user_age_df["age_group"].fillna("xx-24", inplace=True)
    user_age_df = aggregate_duplicate_ids(user_age_df, 'age_group')
    df_results = pd.merge(df_results, user_age_df, on='userid')
    return df_results.drop(columns='age')


def compute_personality(test_data_path, df_results):
    model_path = os.path.join(abs_path, os.path.join("resources", "LinearReg_Personality.sav"))
    profile_df = Utils.read_data_to_dataframe(test_data_path + "/Profile/Profile.csv")
    profile_df.drop(profile_df.columns.difference(['userid', 'ope', 'con', 'ext', 'agr', 'neu']), 1, inplace=True)
    nrc_df = Utils.read_data_to_dataframe(test_data_path + "/Text/nrc.csv")
    liwc_df = Utils.read_data_to_dataframe(test_data_path + "/Text/liwc.csv")

    nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
    liwc_df.rename(columns={'userId': 'userid'}, inplace=True)

    merged_df = pd.merge(nrc_df, liwc_df, on='userid')
    merged_df = pd.merge(merged_df, profile_df, on='userid')

    merged_df.drop(['userid', 'ope', 'con', 'ext', 'agr', 'neu'], axis=1, inplace=True)

    merged_df = np.log(merged_df + 1)
    merged_df = (merged_df - merged_df.min()) / (merged_df.max() - merged_df.min())

    model = Utils.read_pickle_from_file(model_path)
    predictions = model.predict(merged_df)

    nrc_df['ope'] = predictions[:, 0]
    nrc_df['con'] = predictions[:, 1]
    nrc_df['ext'] = predictions[:, 2]
    nrc_df['agr'] = predictions[:, 3]
    nrc_df['neu'] = predictions[:, 4]

    predicted_df = profile_df["userid"].to_frame()
    predicted_df = pd.merge(predicted_df, nrc_df, on="userid", how="left")
    user_pers_df = predicted_df.filter(['userid', 'ope', 'con', 'ext', 'agr', 'neu'])

    df_results.drop(['ope', 'con', 'ext', 'agr', 'neu'], axis=1, inplace=True)
    df_results = pd.merge(df_results, user_pers_df, on='userid', how="left")

    return df_results


def compute_ext(test_data_path, df_results):
    model_path = os.path.join(abs_path, os.path.join("resources", "LinearRegression_ext_v2.sav"))
    profile_df = Utils.read_data_to_dataframe(test_data_path + "/Profile/Profile.csv")
    profile_df.drop(profile_df.columns.difference(['userid', 'ext']), 1, inplace=True)
    nrc_df = Utils.read_data_to_dataframe(test_data_path + "/Text/nrc.csv")
    liwc_df = Utils.read_data_to_dataframe(test_data_path + "/Text/liwc.csv")


    nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
    liwc_df.rename(columns={'userId': 'userid'}, inplace=True)

    merged_df = pd.merge(nrc_df, liwc_df, on='userid')
    merged_df = pd.merge(merged_df, profile_df, on='userid')
    merged_df.drop(['userid', 'ext'], axis=1, inplace=True)
    merged_df = merged_df.filter(
        ['positive', 'negative', 'anger_x', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust',
         'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'future', 'affect', 'posemo', 'negemo', 'anx',
         'incl', 'work', 'death', 'assent', 'nonfl', 'Quote', 'Apostro', 'ext'], axis=1)


    merged_df = np.log(merged_df + 1)
    merged_df = (merged_df - merged_df.min()) / (merged_df.max() - merged_df.min())

    merged_df.fillna(0, inplace=True)

    model = Utils.read_pickle_from_file(model_path)
    predictions = model.predict(merged_df)

    nrc_df['ext'] = predictions[:, 0]

    predicted_df = profile_df["userid"].to_frame()
    predicted_df = pd.merge(predicted_df, nrc_df, on="userid", how="left")
    user_pers_df = predicted_df.filter(['userid', 'ext'])

    df_results = pd.merge(df_results, user_pers_df, on='userid', how="left")
    df_results["ext"] = np.where(df_results['ext_y'].isnull(), df_results['ext_x'], df_results['ext_y'])
    df_results.drop(['ext_x', 'ext_y'], axis=1, inplace=True)

    return df_results


def compute_neu(test_data_path, df_results):
    model_path = os.path.join(abs_path, os.path.join("resources", "RidgeCV_neu.sav"))
    profile_df = Utils.read_data_to_dataframe(test_data_path + "/Profile/Profile.csv")
    profile_df.drop(profile_df.columns.difference(['userid', 'neu']), 1, inplace=True)
    nrc_df = Utils.read_data_to_dataframe(test_data_path + "/Text/nrc.csv")
    liwc_df = Utils.read_data_to_dataframe(test_data_path + "/Text/liwc.csv")
    image_df = read_image(profiles_path=test_data_path + "/Profile/Profile.csv",
                          image_path=test_data_path + "/Image/oxford.csv")

    nrc_df.rename(columns={'userId': 'userid'}, inplace=True)
    liwc_df.rename(columns={'userId': 'userid'}, inplace=True)
    image_df.rename(columns={'userId': 'userid'}, inplace=True)

    merged_df = pd.merge(nrc_df, liwc_df, on='userid')
    merged_df = pd.merge(merged_df, image_df, on='userid')
    merged_df = pd.merge(merged_df, profile_df, on='userid')

    merged_df.drop(['userid', 'neu'], axis=1, inplace=True)

    merged_df = np.log(merged_df + 1)
    merged_df = (merged_df - merged_df.min()) / (merged_df.max() - merged_df.min())

    merged_df.fillna(0, inplace=True)

    model = Utils.read_pickle_from_file(model_path)
    predictions = model.predict(merged_df)

    image_df['neu'] = predictions[:, 0]

    predicted_df = profile_df["userid"].to_frame()
    predicted_df = pd.merge(predicted_df, image_df, on="userid", how="left")
    user_pers_df = predicted_df.filter(['userid', 'neu'])
    user_pers_df = aggregate_duplicate_ids_average(user_pers_df, "neu")

    df_results = pd.merge(df_results, user_pers_df, on='userid', how="left")
    df_results["neu"] = np.where(df_results['neu_y'].isnull(), df_results['neu_x'], df_results['neu_y'])
    df_results.drop(['neu_x', 'neu_y'], axis=1, inplace=True)

    return df_results

def aggregate_duplicate_ids(df, field_name):
    return df.groupby('userid', as_index=False)[field_name].agg(lambda x: x.value_counts().index[0])


def aggregate_duplicate_ids_average(df, field_name):
    return df.groupby('userid', as_index=False)[field_name].agg(lambda x: np.average(x))


class ResultGenerator:
    utils = Utils()

    def generate_results(self, test_data_path="../data/Public_Test/", path_to_results="../data/results"):
        """
        This method Run the test data against model/s and generated XML files
        """
        profiles_path = os.path.join(os.path.join(os.path.join(test_data_path, "Profile")), "Profile.csv")
        profiles = pd.read_csv(profiles_path)
        model_path = os.path.join(abs_path, os.path.join("resources", "model.json"))
        model = self.utils.read_json(model_path)
        df_results = generate_df_for_all_users(profiles, model)

        df_results = compute_gender(test_data_path, df_results)
        df_results = compute_age(test_data_path, df_results)
        df_results = compute_personality(test_data_path, df_results)
        df_results = compute_ext(test_data_path, df_results)
        df_results = compute_neu(test_data_path, df_results)

        xml_dictionary = self.generate_xml_from_profiles(df_results)
        self.store_individual_xmls_into_results_path(path_to_results, xml_dictionary)

    @staticmethod
    def generate_xml_from_profiles(data_frame):
        """
      TODO this should be fixed
        """
        xml_dictionary = {}
        for index, row in data_frame.iterrows():
            xml = "<user \n id = \"" + row["userid"] + "\" " \
                                                       "\n age_group = \"" + str(
                row["age_group"]) + "\" \n gender = \"" + \
                  str(row["gender"]) + "\" \n extrovert = \"" + str(
                row["ext"]) + "\" \n neurotic = \"" + str(
                row["neu"]) + "\" \n agreeable = \"" + str(
                row["agr"]) + "\" \n conscientious = \"" + str(
                row["con"]) + "\" \n open = \"" + str(row["ope"]) + "\" />"
            xml_dictionary[row["userid"]] = xml

        return xml_dictionary

    def store_individual_xmls_into_results_path(self, path_to_results, xml_dictionary):
        """
        This method writes content of a dictionary into files choosing key as the nam eof the file
        """
        self.utils.make_directory_if_not_exists(path_to_results)
        for user in xml_dictionary:
            self.utils.write_to_directory(os.path.join(path_to_results, user + ".xml"), xml_dictionary[user])


if __name__ == "__main__":
    ResultGenerator().generate_results(test_data_path="../data/Public_Test/")
