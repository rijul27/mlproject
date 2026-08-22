import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


# ============================================================
# PREDICTION PIPELINE
# ============================================================
#
# PURPOSE:
# This class loads the trained model and preprocessor
# and uses them to make predictions on new data.
#
# FLOW:
#
# New User Data
#       ↓
# CustomData
#       ↓
# Pandas DataFrame
#       ↓
# Preprocessor
#       ↓
# Transformed Features
#       ↓
# Trained Model
#       ↓
# Prediction
#
# Saved objects used:
#
#     artifacts/preprocessor.pkl
#     artifacts/model.pkl
#
# ============================================================


class PredictPipeline:

    def __init__(self):
        """
        Constructor for PredictPipeline.

        No initialization is required because the model
        and preprocessor are loaded when prediction starts.
        """
        pass

    # ========================================================
    # PREDICT
    # ========================================================
    #
    # PURPOSE:
    # Transform new input data using the saved preprocessor
    # and generate predictions using the saved model.
    #
    # ========================================================

    def predict(self, features):

        try:

            # =================================================
            # STEP 1: DEFINE MODEL PATH
            # =================================================
            #
            # The trained model was saved during model training.
            #
            # Location:
            #
            #     artifacts/model.pkl
            #
            # =================================================

            model_path = os.path.join("artifacts","model.pkl")

            # =================================================
            # STEP 2: DEFINE PREPROCESSOR PATH
            # =================================================
            #
            # The preprocessor was created during Data
            # Transformation and saved as:
            #
            #     artifacts/preprocessor.pkl
            #
            # We must use the same preprocessor during
            # prediction.
            #
            # =================================================

            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

            # =================================================
            # STEP 3: LOAD TRAINED MODEL
            # =================================================
            #
            # load_object() loads the model that was saved
            # using pickle.
            #
            # We don't train the model again.
            #
            # =================================================

            model = load_object(file_path=model_path)

            # =================================================
            # STEP 4: LOAD PREPROCESSOR
            # =================================================
            #
            # The preprocessor contains the transformations
            # learned during training.
            #
            # For example:
            #
            # Numerical features:
            #     - Missing value handling
            #     - StandardScaler
            #
            # Categorical features:
            #     - Missing value handling
            #     - OneHotEncoder
            #     - Scaling
            #
            # =================================================

            preprocessor = load_object(file_path=preprocessor_path)

            # =================================================
            # STEP 5: TRANSFORM NEW INPUT DATA
            # =================================================
            #
            # The new user data is still in its original form.
            #
            # Example:
            #
            # gender = male
            # lunch = standard
            # reading_score = 70
            #
            # The saved preprocessor converts this data into
            # the same format that the model saw during training.
            #
            # IMPORTANT:
            #
            # Use transform()
            #
            # NOT:
            #
            # fit_transform()
            #
            # Because the preprocessor was already fitted
            # using the training data.
            #
            # =================================================

            data_scaled = preprocessor.transform(features)

            # =================================================
            # STEP 6: MAKE PREDICTION
            # =================================================
            #
            # Pass the transformed data to the trained model.
            #
            # Example:
            #
            # transformed features
            #          ↓
            #      trained model
            #          ↓
            #    predicted score
            #
            # =================================================

            preds = model.predict(data_scaled)

            # =================================================
            # STEP 7: RETURN PREDICTION
            # =================================================

            return preds


        except Exception as e:

            # If anything goes wrong, raise our custom
            # exception with file name and line number.

            raise CustomException(
                e,
                sys
            )


# ============================================================
# CUSTOM DATA
# ============================================================
#
# PURPOSE:
# This class receives the values entered by the user and
# converts them into a Pandas DataFrame.
#
# The column names must match the columns used during
# model training.
#
# ============================================================


class CustomData:

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):

        # Store the user input.

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = (parental_level_of_education)
        self.lunch = lunch
        self.test_preparation_course = (test_preparation_course)
        self.reading_score = reading_score
        self.writing_score = writing_score

    # ========================================================
    # GET DATA AS DATAFRAME
    # ========================================================
    #
    # PURPOSE:
    # Convert the user's input into a Pandas DataFrame.
    #
    # ========================================================

    def get_data_as_data_frame(self):

        try:
            # =================================================
            # STEP 1: CREATE INPUT DICTIONARY
            # =================================================
            #
            # The column names here must be exactly the same
            # as the feature names used during training.
            #
            # =================================================

            custom_data_input_dict = {

                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }


            # =================================================
            # STEP 2: CREATE DATAFRAME
            # =================================================
            #
            # Convert the dictionary into a Pandas DataFrame.
            #
            # This DataFrame is passed to PredictPipeline.
            #
            # =================================================

            custom_data_input_df = pd.DataFrame(custom_data_input_dict)

            return custom_data_input_df

        except Exception as e:
            raise CustomException(
                e,
                sys
            )