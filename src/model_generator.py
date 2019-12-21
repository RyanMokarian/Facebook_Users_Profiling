"""
This file contains scripts to generate different models
"""
from sklearn.linear_model import LogisticRegression

from src.classifiers.combined_classifier import CombinedClassifier
from src.classifiers.image_classifier import ImageClassifier
from src.preprocessors.nrc import Personality


def generate_models():
    # generate model for age prediction
    clf = LogisticRegression(C=0.004832930238571752, penalty='l2')
    CC = CombinedClassifier()
    df = CC.merge_images_piwc()
    CC.predict_age_using_logistic_regression(df)

    # generate models for personality prediction
    personality = Personality()
    personality.generate_all_personality_model()
    personality.generate_ext_model()  # add ext model
    personality.generate_neu_model()  # add neu model
    # generate model for gender prediction
    image_classifier = ImageClassifier()
    df_gender = image_classifier.get_image_gender_training_data()
    image_classifier.generate_model_using_random_forest_classifier(df_gender)


if __name__ == "__main__":
    generate_models()
