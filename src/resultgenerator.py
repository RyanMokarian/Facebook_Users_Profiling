import os
import pickle
import pandas as pd

from src.util import Utils

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
    predicted_df = profile_df["userid"]
    predicted_df = pd.merge(predicted_df, image_df, on='userid', how="left")
    user_gender_df = predicted_df.filter(["userid", "gender"])
    user_gender_df["gender"].fillna(0, inplace=True)
    pd.merge(df_results, user_gender_df, on='userid')
    return user_gender_df


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
    pass


def compute_personality(test_data_path, df_results):
    pass


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

        xml_dictionary = self.generate_xml_from_profiles(profiles, df_results)
        self.store_individual_xmls_into_results_path(path_to_results, xml_dictionary, )

    @staticmethod
    def generate_xml_from_profiles(profiles, data_frame):
        """
      TODO this should be fixed
        """
        gender_map = {1: "male", 2: "female"}
        xml_dictionary = {}
        for profile in profiles:
            xml = "<user \n id = \"" + profile[1] + "\" " \
                                                    "\n age_group = \"" + str(
                data_frame["age_group"]) + "\" \n gender = \"" + \
                  str(gender_map[
                          data_frame.loc(data_frame['userid'] == profile[1])["gender"]]) + "\" \n extrovert = \"" + str(
                data_frame["extrovert"]) + "\" \n neurotic = \"" + str(
                data_frame[
                    "neurotic"]) + "\" \n agreeable = \"" + str(
                data_frame["agreeable"]) + "\" \n conscientious = \"" + str(
                data_frame[
                    "conscientious"]) + "\" \n open = \"" + str(data_frame[
                                                                    "open"]) + "\" />"
            xml_dictionary[profile[1]] = xml

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
