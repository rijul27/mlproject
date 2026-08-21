import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score


# ============================================================
# SAVE OBJECT FUNCTION
# ============================================================
# This function is used to save any Python object to a file.
#
# In our ML project, we use this function to save objects such
# as:
#
#     preprocessing object → preprocessor.pkl
#     trained model        → model.pkl
#
# We use Python's pickle module to save the object.
# ============================================================

def save_object(file_path, obj):

    """
    Save a Python object to a file using pickle.

    Parameters
    ----------
    file_path : str
        Location where the object should be saved.

    obj : object
        Python object that we want to save.

    Example
    -------
    save_object(
        "artifacts/preprocessor.pkl",
        preprocessing_obj
    )
    """

    try:

        # ====================================================
        # STEP 1: GET DIRECTORY PATH
        # ====================================================
        # Suppose file_path is:
        #
        # artifacts/preprocessor.pkl
        #
        # os.path.dirname() gives:
        #
        # artifacts
        # ====================================================

        dir_path = os.path.dirname(file_path)


        # ====================================================
        # STEP 2: CREATE DIRECTORY
        # ====================================================
        # Create the directory if it doesn't already exist.
        #
        # exist_ok=True means:
        #
        # Folder exists     → don't throw an error
        # Folder doesn't    → create the folder
        # ====================================================

        os.makedirs(dir_path, exist_ok=True)


        # ====================================================
        # STEP 3: OPEN FILE
        # ====================================================
        # "wb" means:
        #
        # w → write
        # b → binary
        #
        # Pickle objects are stored in binary format.
        # ====================================================

        with open(file_path, "wb") as file_obj:


            # =================================================
            # STEP 4: SAVE OBJECT USING PICKLE
            # =================================================
            # pickle.dump() converts the Python object into a
            # binary format and saves it inside the file.
            # =================================================

            pickle.dump(obj, file_obj)


    # ========================================================
    # STEP 5: HANDLE ERRORS
    # ========================================================
    # If anything goes wrong while creating the directory,
    # opening the file, or saving the object, our custom
    # exception will provide detailed error information.
    # ========================================================

    except Exception as e:

        raise CustomException(e, sys)


# ============================================================
# EVALUATE MULTIPLE MODELS
# ============================================================
# This function trains multiple machine learning models
# and compares their performance.
#
# Input:
#
#     X_train → training features
#     y_train → training target
#     X_test  → testing features
#     y_test  → testing target
#     models  → dictionary containing ML models
#
# Output: Dictionary containing:
#
#     model name → R² score
#
# Example:
#
# {
#     "Linear Regression": 0.87,
#     "Random Forest": 0.91,
#     "XGBRegressor": 0.89
# }
#
# ============================================================
def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:
        # ----------------------------------------------------
        # Create an empty dictionary.
        #
        # We will store the performance of every model here.
        # ----------------------------------------------------
        model_report = {}

        # ====================================================
        # LOOP THROUGH ALL MODELS
        # ====================================================
        #
        # models looks something like:
        #
        # {
        #     "Linear Regression": LinearRegression(),
        #     "Random Forest": RandomForestRegressor(),
        #     "XGBRegressor": XGBRegressor()
        # }
        #
        # Each loop gives us:
        #
        # model_name → "Linear Regression"
        #
        # model      → LinearRegression()
        # ====================================================

        for model_name, model in models.items():
            # Fit the model on the training data
            # The model learns the relationship between:
            #
            # X_train → input features
            # y_train → actual target
            #
            # Example:
            #
            # reading_score
            # writing_score
            # gender
            # ...
            #
            #          ↓
            #
            #       MODEL
            #
            #          ↓
            #
            #      math_score
            model.fit(X_train, y_train)

            # Predict on the test data
            # Use the trained model to predict the target
            # values for data it has not seen during training.
            #
            # X_test → unseen testing features
            #
            # y_pred → model predictions
            y_pred = model.predict(X_test)

            # Calculate R^2 score and store it in the report
            # Compare:
            #
            # y_test → actual values
            #
            # y_pred → predicted values
            #
            # R² score tells us how well the model explains
            # the variation in the target variable.
            #
            # Higher R² is generally better.
            score = r2_score(y_test, y_pred)

            # Store the model name and its R² score
            model_report[model_name] = score

        # Return the complete report after evaluating
        # every model.
        return model_report

    except Exception as e:
        # If any model fails during training, prediction,
        # or evaluation, raise our custom exception.
        raise CustomException(e, sys)