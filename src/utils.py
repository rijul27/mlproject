import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
#
# PURPOSE:
# This function performs hyperparameter tuning for the models
# for which we have provided a parameter grid.
#
# Hyperparameters are settings that we choose BEFORE training
# a machine learning model.
#
# Example:
#
# Random Forest:
#
#     n_estimators = 100
#     max_depth = 10
#
# Instead of manually trying different values, we use
# GridSearchCV to automatically try different combinations.
#
#
# INPUT:
#
#     X_train
#         Training features
#
#     y_train
#         Training target
#
#     X_test
#         Testing features
#
#     y_test
#         Testing target
#
#     models
#         Dictionary containing all machine learning models
#
#     param_grids
#         Dictionary containing hyperparameter combinations
#         for the models that we want to tune.
#
#
# OUTPUT:
#
#     tuned_model_report
#         Model name → R² score on test data
#
#     tuned_models
#         Model name → best trained model
#
#     best_params
#         Model name → best hyperparameters
#
#
# IMPORTANT:
#
# We may have 8 models in our `models` dictionary, but we don't
# necessarily want to tune all 8 models.
#
# Therefore, this function only tunes models that are present
# inside `param_grids`.
#
# ============================================================

def tune_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param_grids
):

    try:

        # ====================================================
        # CREATE RESULT DICTIONARIES
        # ====================================================

        # Stores:
        #
        # model name → R² score
        #
        # Example:
        #
        # {
        #     "Random Forest": 0.91,
        #     "XGBRegressor": 0.93
        # }

        tuned_model_report = {}

        # Stores the best trained model for each tuned model.

        tuned_models = {}

        # Stores the best hyperparameters found by GridSearchCV.

        best_params = {}

        # ====================================================
        # LOOP THROUGH ALL MODELS
        # ====================================================
        #
        # The `models` dictionary may contain:
        #
        #     Linear Regression
        #     KNN
        #     Decision Tree
        #     Random Forest
        #     XGBoost
        #     CatBoost
        #     AdaBoost
        #     Gradient Boosting
        #
        # But we only want to tune models for which we have
        # defined a hyperparameter grid.
        #
        # ====================================================

        for model_name, model in models.items():


            # =================================================
            # CHECK WHETHER THIS MODEL SHOULD BE TUNED
            # =================================================
            #
            # Example:
            #
            # param_grids contains:
            #
            #     Random Forest
            #     XGBRegressor
            #     CatBoosting Regressor
            #     Gradient Boosting Regressor
            #
            # If the current model isn't present in
            # param_grids, skip it.
            #
            # =================================================

            if model_name not in param_grids:

                continue

            # =================================================
            # GET THE HYPERPARAMETER GRID
            # =================================================
            #
            # Example:
            #
            # For Random Forest:
            #
            # {
            #     "n_estimators": [100, 200],
            #     "max_depth": [None, 10, 20]
            # }
            #
            # =================================================

            params = param_grids[
                model_name
            ]


            # =================================================
            # CREATE GRIDSEARCHCV
            # =================================================
            #
            # GridSearchCV automatically tries different
            # combinations of hyperparameters.
            #
            # Example:
            #
            # n_estimators:
            #     [100, 200]
            #
            # max_depth:
            #     [10, 20]
            #
            # GridSearchCV will try:
            #
            #     100 + 10
            #     100 + 20
            #     200 + 10
            #     200 + 20
            #
            # cv=5:
            #
            # The training data is divided into 5 parts.
            # Cross-validation is performed using those parts.
            #
            # scoring="r2":
            #
            # R² score is used to decide which parameter
            # combination performs best.
            #
            # n_jobs=-1:
            #
            # Use all available CPU cores to speed up the search.
            #
            # =================================================

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )


            # =================================================
            # TRAIN GRID SEARCH
            # =================================================
            #
            # GridSearchCV now:
            #
            #     1. Tries every parameter combination
            #     2. Performs 5-fold cross-validation
            #     3. Calculates R²
            #     4. Finds the best combination
            #
            # =================================================

            logging.info(
                f"Starting hyperparameter tuning for: {model_name}"
            )

            grid_search.fit(
                X_train,
                y_train
            )


            # =================================================
            # GET BEST MODEL
            # =================================================
            #
            # best_estimator_ gives us the model trained using
            # the best hyperparameter combination.
            #
            # Example:
            #
            # Random Forest
            #
            # Before:
            #
            #     n_estimators = default
            #
            # After GridSearch:
            #
            #     n_estimators = 200
            #     max_depth = 10
            #
            # =================================================

            best_model = grid_search.best_estimator_


            # =================================================
            # GET BEST PARAMETERS
            # =================================================
            #
            # best_params_ contains the hyperparameter
            # combination that produced the best CV score.
            #
            # Example:
            #
            # {
            #     "max_depth": 10,
            #     "n_estimators": 200
            # }
            #
            # =================================================

            best_params[
                model_name
            ] = grid_search.best_params_


            # =================================================
            # PREDICT USING THE BEST MODEL
            # =================================================
            #
            # Now that GridSearchCV has found the best model,
            # use it to make predictions on the test dataset.
            #
            # =================================================

            y_pred = best_model.predict(
                X_test
            )


            # =================================================
            # CALCULATE TEST R² SCORE
            # =================================================
            #
            # Compare:
            #
            # y_test
            #     ↓
            # Actual target values
            #
            # y_pred
            #     ↓
            # Predicted target values
            #
            # This tells us how well the tuned model performs
            # on our test dataset.
            #
            # =================================================

            score = r2_score(
                y_test,
                y_pred
            )


            # =================================================
            # STORE THE TUNED MODEL SCORE
            # =================================================

            tuned_model_report[
                model_name
            ] = score


            # =================================================
            # STORE THE BEST TRAINED MODEL
            # =================================================
            #
            # We save the model so that ModelTrainer can later
            # compare all tuned models and select the overall
            # best model.
            #
            # =================================================

            tuned_models[
                model_name
            ] = best_model


            # =================================================
            # LOG RESULTS
            # =================================================

            logging.info(
                f"Completed tuning for {model_name}. "
                f"Best parameters: {grid_search.best_params_}. "
                f"Test R2 score: {score}"
            )


        # ====================================================
        # RETURN ALL TUNING RESULTS
        # ====================================================
        #
        # Return:
        #
        # 1. tuned_model_report
        #    → Model performance
        #
        # 2. tuned_models
        #    → Best trained models
        #
        # 3. best_params
        #    → Best hyperparameters
        #
        # ====================================================

        return (
            tuned_model_report,
            tuned_models,
            best_params
        )


    # ========================================================
    # EXCEPTION HANDLING
    # ========================================================

    except Exception as e:

        raise CustomException(
            e,
            sys
        )

# ==================================================================
# Code without Hyperparameters Tunning
# ==================================================================


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

# def save_object(file_path, obj):

#     """
#     Save a Python object to a file using pickle.

#     Parameters
#     ----------
#     file_path : str
#         Location where the object should be saved.

#     obj : object
#         Python object that we want to save.

#     Example
#     -------
#     save_object(
#         "artifacts/preprocessor.pkl",
#         preprocessing_obj
#     )
#     """

#     try:

#         # ====================================================
#         # STEP 1: GET DIRECTORY PATH
#         # ====================================================
#         # Suppose file_path is:
#         #
#         # artifacts/preprocessor.pkl
#         #
#         # os.path.dirname() gives:
#         #
#         # artifacts
#         # ====================================================

#         dir_path = os.path.dirname(file_path)


#         # ====================================================
#         # STEP 2: CREATE DIRECTORY
#         # ====================================================
#         # Create the directory if it doesn't already exist.
#         #
#         # exist_ok=True means:
#         #
#         # Folder exists     → don't throw an error
#         # Folder doesn't    → create the folder
#         # ====================================================

#         os.makedirs(dir_path, exist_ok=True)


#         # ====================================================
#         # STEP 3: OPEN FILE
#         # ====================================================
#         # "wb" means:
#         #
#         # w → write
#         # b → binary
#         #
#         # Pickle objects are stored in binary format.
#         # ====================================================

#         with open(file_path, "wb") as file_obj:


#             # =================================================
#             # STEP 4: SAVE OBJECT USING PICKLE
#             # =================================================
#             # pickle.dump() converts the Python object into a
#             # binary format and saves it inside the file.
#             # =================================================

#             pickle.dump(obj, file_obj)


#     # ========================================================
#     # STEP 5: HANDLE ERRORS
#     # ========================================================
#     # If anything goes wrong while creating the directory,
#     # opening the file, or saving the object, our custom
#     # exception will provide detailed error information.
#     # ========================================================

#     except Exception as e:

#         raise CustomException(e, sys)


# # ============================================================
# # EVALUATE MULTIPLE MODELS
# # ============================================================
# # This function trains multiple machine learning models
# # and compares their performance.
# #
# # Input:
# #
# #     X_train → training features
# #     y_train → training target
# #     X_test  → testing features
# #     y_test  → testing target
# #     models  → dictionary containing ML models
# #
# # Output: Dictionary containing:
# #
# #     model name → R² score
# #
# # Example:
# #
# # {
# #     "Linear Regression": 0.87,
# #     "Random Forest": 0.91,
# #     "XGBRegressor": 0.89
# # }
# #
# # ============================================================
# def evaluate_models(X_train, y_train, X_test, y_test, models):

#     try:
#         # ----------------------------------------------------
#         # Create an empty dictionary.
#         #
#         # We will store the performance of every model here.
#         # ----------------------------------------------------
#         model_report = {}

#         # ====================================================
#         # LOOP THROUGH ALL MODELS
#         # ====================================================
#         #
#         # models looks something like:
#         #
#         # {
#         #     "Linear Regression": LinearRegression(),
#         #     "Random Forest": RandomForestRegressor(),
#         #     "XGBRegressor": XGBRegressor()
#         # }
#         #
#         # Each loop gives us:
#         #
#         # model_name → "Linear Regression"
#         #
#         # model      → LinearRegression()
#         # ====================================================

#         for model_name, model in models.items():
#             # Fit the model on the training data
#             # The model learns the relationship between:
#             #
#             # X_train → input features
#             # y_train → actual target
#             #
#             # Example:
#             #
#             # reading_score
#             # writing_score
#             # gender
#             # ...
#             #
#             #          ↓
#             #
#             #       MODEL
#             #
#             #          ↓
#             #
#             #      math_score
#             model.fit(X_train, y_train)

#             # Predict on the test data
#             # Use the trained model to predict the target
#             # values for data it has not seen during training.
#             #
#             # X_test → unseen testing features
#             #
#             # y_pred → model predictions
#             y_pred = model.predict(X_test)

#             # Calculate R^2 score and store it in the report
#             # Compare:
#             #
#             # y_test → actual values
#             #
#             # y_pred → predicted values
#             #
#             # R² score tells us how well the model explains
#             # the variation in the target variable.
#             #
#             # Higher R² is generally better.
#             score = r2_score(y_test, y_pred)

#             # Store the model name and its R² score
#             model_report[model_name] = score

#         # Return the complete report after evaluating
#         # every model.
#         return model_report

#     except Exception as e:
#         # If any model fails during training, prediction,
#         # or evaluation, raise our custom exception.
#         raise CustomException(e, sys)