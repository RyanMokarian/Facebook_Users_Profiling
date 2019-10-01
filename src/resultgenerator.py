from util import Utils
import os


class ResultGenerator:
    utils = Utils()

    def generate_results(self, test_data="../data/Public_Test/", path_to_results="../data/results"):
        """
        This method Run the test data against model/s and generated XML files
        """
        profiles_path = os.path.join(os.path.join(os.path.join(test_data, "Profile")), "Profile.csv")
        profiles = self.utils.read_csv(profiles_path)
        profiles.pop(0)
        abs_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(abs_path, os.path.join("resources", "model.json"))
        model = self.utils.read_json(model_path)
        xml_dictionary = self.generate_xml_from_profiles(profiles, model)
        self.store_individual_xmls_into_results_path(path_to_results, xml_dictionary, )

    @staticmethod
    def generate_xml_from_profiles(profiles, model):
        """
        This method iterates through profiles and generates a dictionary of
        Key: user_id
        Value: XML element having user attributes
        """
        xml_dictionary = {}
        for profile in profiles:
            xml = "<user \n id = \"" + profile[1] + "\" " \
                                                    "\n age_group = \"" + str(
                model["age_group"]) + "\" \n gender = \"" + \
                  str(model["gender"]) + "\" \n extrovert = \"" + str(model["extrovert"]) + "\" \n neurotic = \"" + str(
                model[
                    "neurotic"]) + "\" \n agreeable = \"" + str(model["agreeable"]) + "\" \n conscientious = \"" + str(
                model[
                    "conscientious"]) + "\" \n open = \"" + str(model[
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
