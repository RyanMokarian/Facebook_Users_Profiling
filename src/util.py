import json
import os


class Utils:

    def read_json(self, filename):
        """
        This method loads a json from a file
        :param filename:
        :return: a json file
        """
        full_path = os.path.join("resources", filename)
        resources = {}
        if full_path:
            with open(full_path, 'r') as file:
                resources = json.load(file)
        return resources

    def write_json_to_directory(self, json_object, file_name, directory="resources"):
        """
        This method writes a python object to a json file
        """
        self.make_directory_if_not_exists(directory)
        filename = os.path.join(directory, file_name )
        with open(filename, 'w') as outfile:
            json.dump(json_object, outfile)

    @staticmethod
    def make_directory_if_not_exists(directory):
        """
        this method creates a directory if it does not exists
        :param directory:
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
