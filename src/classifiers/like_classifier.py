from src.util import Utils
import pandas as pd


class LikeClassifier:

    @staticmethod
    def generate_data():
        util = Utils()
        relation_df = util.read_data_to_dataframe("../../data/Train/Profile/Profile.csv")
        profile_df = util.read_data_to_dataframe("../../data/Train/Relation/Relation.csv")
        merged_df = pd.merge(relation_df, profile_df, on='userid')
        merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
        return merged_df.filter(['like_id', 'gender'], axis=1)


if __name__ == "__main__":
    LikeClassifier().generate_data()
