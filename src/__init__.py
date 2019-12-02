import sys

from src.resultgenerator import ResultGenerator


def main(test_data="data/Public_Test/Profile/Profile.csv", path_to_results="data/results"):
    # TODO Running the model  against the test data and save them in result path
    ResultGenerator().generate_results(test_data, path_to_results)
    pass


if __name__ == "__main__":
    test_data = sys.argv[1]
    results_path = sys.argv[2]
    main(test_data, results_path)
