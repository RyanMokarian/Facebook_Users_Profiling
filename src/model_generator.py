"""
This file contains scripts to generate different models
"""
from classifiers.combined_classifier import CombinedClassifier
from classifiers.image_classifier import ImageClassifier
from classifiers.nrc import Personality


def generate_models():
    # generate model for age prediction
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
