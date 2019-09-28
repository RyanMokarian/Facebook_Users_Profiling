from src.util import Utils
import os
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


class ResultGenerator:
    utils = Utils()

    def generate_results(self, test_data="../data/Public_Test/", path_to_results="../data/results"):
        profiles_path = os.path.join(os.path.join(os.path.join(test_data, "Profile")), "Profile.csv")
        profiles = self.utils.read_csv(profiles_path)
        profiles.pop(0)
        model_path = os.path.join("resources", "model.json")
        model = self.utils.read_json(model_path)
        list_of_xmls = self.generate_xml_from_profiles(profiles, model)
        self.store_individual_xmls_into_results_path(list_of_xmls, path_to_results, profiles)

    def generate_xml_from_profiles(self, profiles, model):
        xml_list = []
        for profile in profiles:
            xml_list.append(
                "< user id = \"" + profile[1] + "\" age_group = \"" + model["age_group"] + "\" gender = \"" +
                model["gender"] + "\" extrovert = \"" + model["extrovert"] + "\" neurotic = \"" + model[
                    "neurotic"] + "\" agreeable = \"" +
                model["agreeable"] + "\" conscientious = \"" + model["conscientious"] + "\" open = \"" + model[
                    "open"] + "\" / >")

        return xml_list

    def store_individual_xmls_into_results_path(self, list_of_xmls, path_to_results, profiles):
        pass

    # < user
    # id = "8157f43c71fbf53f4580fd3fc808bd29"
    # age_group = "xx-24"
    # gender = "female"
    # extrovert = "2.7"
    # neurotic = "4.55"
    # 1
    # agreeable = "3"
    # conscientious = "1.9"
    # open = "2.1"
    # / >
