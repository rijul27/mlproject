import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from src.utils import (
    save_object,
    evaluate_models,
    tune_models
)

# ============================================================
# MODEL TRAINER CONFIGURATION
# ============================================================
# This class stores the configuration required by the
# ModelTrainer component.
#
# The best trained model will be saved here:
#
# artifacts/model.pkl
# ============================================================

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# ================================================================================================
# MODEL TRAINER
# ================================================================================================
#
# PURPOSE:
# This class is responsible for:
#
#     1. Separating features (X) and target (y)
#     2. Creating multiple ML models
#     3. Evaluating models with default parameters
#     4. Performing hyperparameter tuning
#     5. Comparing tuned models
#     6. Selecting the best model
#     7. Saving the best model
#
# ================================================================================================
class ModelTrainer:
    def __init__(self):

        # Create ModelTrainerConfig object.
        #
        # This gives us access to:
        #
        # self.model_trainer_config.trained_model_file_path

        self.model_trainer_config = ModelTrainerConfig()

    # ========================================================
    # INITIATE MODEL TRAINING
    # ========================================================

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # ====================================================================================
            # STEP 1: SEPARATE FEATURES AND TARGET
            # ====================================================================================
            #
            # Data Transformation returns an array in this format:
            #
            #     [feature1, feature2, feature3, ..., target]
            #
            # The LAST column is the target variable:
            #
            #     math_score
            #
            # Therefore:
            #
            #     [:, :-1]
            #         → all rows
            #         → all columns except the last
            #         → input features (X)
            #
            #     [:, -1]
            #         → all rows
            #         → only the last column
            #         → target (y)
            #
            # ====================================================================================

            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                # Training features
                train_array[:, :-1],
                # Training target
                train_array[:, -1],
                # Testing features
                test_array[:, :-1],
                # Testing target
                test_array[:, -1]
            )

            # ====================================================================================
            # STEP 2: CREATE MACHINE LEARNING MODELS
            # ====================================================================================
            #
            # Create multiple regression models.
            #
            # Initially, we use the default hyperparameters.
            #
            # Later, promising models will be tuned using GridSearchCV.
            #
            # ====================================================================================

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor()
            }

            # ====================================================================================
            # STEP 3: BASELINE MODEL EVALUATION
            # ====================================================================================
            #
            # First, evaluate all models using their DEFAULT hyperparameters.
            #
            # This gives us a baseline score.
            #
            # Example:
            #
            #     Linear Regression → 0.87
            #     Random Forest     → 0.89
            #     XGBoost           → 0.91
            #
            # We can later compare these scores with the tuned scores.
            #
            # ====================================================================================

            logging.info("Starting baseline model evaluation")

            model_report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # Display baseline model performance

            print("\n==============================================")
            print("BASELINE MODEL PERFORMANCE")
            print("==============================================")

            for model_name, score in model_report.items():

                print(f"{model_name}: {score:.4f}")

            # ====================================================================================
            # STEP 4: DEFINE HYPERPARAMETER GRIDS
            # ====================================================================================
            #
            # PURPOSE:
            # Define different hyperparameter values that GridSearchCV will try.
            #
            # We are tuning the following models:
            #
            #     - Random Forest
            #     - XGBoost
            #     - CatBoost
            #     - Gradient Boosting
            #
            # The other models are kept with their default parameters for now.
            #
            # ====================================================================================

            param_grids = {

                # -------------------------------------------------------------------------------
                # Random Forest
                # -------------------------------------------------------------------------------

                "Random Forest": {

                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1,2]
                },

                # -------------------------------------------------------------------------------
                # XGBoost
                # -------------------------------------------------------------------------------

                "XGBRegressor": {

                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                },

                # -------------------------------------------------------------------------------
                # CatBoost
                # -------------------------------------------------------------------------------

                "CatBoosting Regressor": {

                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4,6,8]
                },

                # -------------------------------------------------------------------------------
                # Gradient Boosting
                # -------------------------------------------------------------------------------

                "Gradient Boosting Regressor": {

                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7]
                }
            }

            # ====================================================================================
            # STEP 5: HYPERPARAMETER TUNING
            # ====================================================================================
            #
            # Try different combinations of hyperparameters and find the combination that
            # produces the best cross-validation R² score.
            #
            # GridSearchCV performs:
            #
            #     Hyperparameter combination
            #             ↓
            #         Train model
            #             ↓
            #       Cross-validation
            #             ↓
            #         Calculate R²
            #             ↓
            #       Try next combination
            #             ↓
            #          Best parameters
            #
            # The tune_models() function returns:
            #
            #     tuned_model_report
            #         → R² score of each tuned model
            #
            #     tuned_models
            #         → trained models with best parameters
            #
            #     best_params
            #         → best hyperparameters for each model
            #
            # ====================================================================================

            logging.info("Starting hyperparameter tuning")

            (tuned_model_report,tuned_models,best_params) = tune_models(X_train=X_train, y_train=y_train,X_test=X_test,
                                                                         y_test=y_test,models=models,param_grids=param_grids
            )

            # ====================================================================================
            # STEP 6: DISPLAY TUNED MODEL PERFORMANCE
            # ====================================================================================
            #
            # Display the R² score achieved by every tuned model.
            #
            # ====================================================================================

            print("\n==============================================")
            print("TUNED MODEL PERFORMANCE")
            print("==============================================")

            for model_name, score in tuned_model_report.items():

                print(f"{model_name}: {score:.4f}")

            # ====================================================================================
            # STEP 7: DISPLAY BEST HYPERPARAMETERS
            # ====================================================================================
            #
            # Display the hyperparameters selected by GridSearchCV.
            #
            # Example:
            #
            # Random Forest:
            # {
            #     'max_depth': 10,
            #     'n_estimators': 200,
            #     ...
            # }
            #
            # ====================================================================================

            print("\n==============================================")
            print("BEST HYPERPARAMETERS")
            print("==============================================")

            for model_name, params in best_params.items():
                print(f"\n{model_name}:")
                print(params)
            
            # ====================================================================================
            # STEP 8: FIND THE BEST TUNED MODEL SCORE
            # ====================================================================================
            #
            # Find the highest R² score from the tuned models.
            #
            # ====================================================================================

            best_model_score = max(tuned_model_report.values())

            # ====================================================================================
            # STEP 9: FIND THE BEST TUNED MODEL NAME
            # ====================================================================================
            #
            # We now know the highest score.
            #
            # Next, find which model produced that score.
            #
            # Example:
            #
            #     best_model_score = 0.93
            #
            #     best_model_name = "XGBRegressor"
            #
            # ====================================================================================

            best_model_name = list(
                tuned_model_report.keys()
            )[
                list(
                    tuned_model_report.values()
                ).index(
                    best_model_score
                )
            ]


            # ====================================================================================
            # STEP 10: GET THE BEST MODEL OBJECT
            # ====================================================================================
            #
            # tuned_models contains the models after GridSearchCV has selected their
            # best hyperparameters.
            #
            # Get the model corresponding to the highest tuned score.
            #
            # ====================================================================================

            best_model = tuned_models[best_model_name]

            # ====================================================================================
            # STEP 11: CHECK MINIMUM MODEL PERFORMANCE
            # ====================================================================================
            #
            # We only accept the model if its R² score is at least 0.60.
            #
            # If:
            #
            #     R² < 0.60
            #
            # then we consider the model performance insufficient.
            #
            # ====================================================================================

            if best_model_score < 0.60:

                raise CustomException(
                    "No best model found"
                )

            # Log the selected model and its performance.

            logging.info(
                f"Best tuned model: {best_model_name} " 
                f"with R2 score: {best_model_score}"
            )

            # ====================================================================================
            # STEP 12: SAVE THE BEST MODEL
            # ====================================================================================
            #
            # Save the selected tuned model as:
            #
            #     artifacts/model.pkl
            #
            # Later, this model can be loaded and used for predictions without training again.
            #
            # ====================================================================================

            save_object(

                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),

                obj=best_model
            )

            # ====================================================================================
            # STEP 13: FINAL TEST SET PREDICTION
            # ====================================================================================
            #
            # Use the selected best model to make predictions on X_test.
            #
            # ====================================================================================

            predicted = best_model.predict(X_test)

            # ====================================================================================
            # STEP 14: CALCULATE FINAL R² SCORE
            # ====================================================================================
            #
            # Compare:
            #
            #     y_test
            #         ↓
            #     Actual values
            #
            #     predicted
            #         ↓
            #     Model predictions
            #
            # R² tells us how well the model explains the variation in the target.
            #
            # ====================================================================================

            r2_square = r2_score(y_test,predicted)

            # Log final score.

            logging.info(f"Final R2 score: {r2_square}")

            # Return the final score.

            return r2_square

        # ========================================================================================
        # EXCEPTION HANDLING
        # ========================================================================================

        except Exception as e:

            raise CustomException(
                e,
                sys
            )




# ==================================================================
# Code without Hyperparameters Tunning
# ==================================================================

# ============================================================
# MODEL TRAINER CONFIGURATION
# ============================================================
# This class stores the configuration required by the
# ModelTrainer component.
#
# The best trained model will be saved here:
#
# artifacts/model.pkl
# ============================================================

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):

#         # Create ModelTrainerConfig object.
#         #
#         # This gives us access to:
#         #
#         # self.model_trainer_config.trained_model_file_path

#         self.model_trainer_config = ModelTrainerConfig()

#     # ========================================================
#     # INITIATE MODEL TRAINING
#     # ========================================================
#     #
#     # This method is responsible for:
#     #
#     # 1. Separating X and y
#     # 2. Creating multiple ML models
#     # 3. Training and evaluating all models
#     # 4. Finding the best model
#     # 5. Checking whether the model is good enough
#     # 6. Saving the best model
#     # 7. Returning the final R² score
#     # ========================================================

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             # =================================================
#             # STEP 1: SEPARATE FEATURES AND TARGET
#             # =================================================
#             #
#             # Data Transformation created arrays like:
#             #
#             # [feature1, feature2, feature3, ..., target]
#             #
#             # The LAST column is our target: math_score
#             #
#             # Therefore:
#             #
#             # [:, :-1]
#             # → all rows, all columns except last
#             # → X
#             #
#             # [:, -1]
#             # → all rows, only the last column
#             # → y
#             # =================================================

#             logging.info("Splitting training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 # Training features
#                 train_array[:, :-1],
#                 # Training target
#                 train_array[:, -1],
#                 # Testing features
#                 test_array[:, :-1],
#                 # Testing target
#                 test_array[:, -1]
#             )

#             # =================================================
#             # STEP 2: CREATE MULTIPLE ML MODELS
#             # =================================================
#             #
#             # We don't know beforehand which algorithm will
#             # perform best on our dataset.
#             #
#             # Therefore, we train several models and compare
#             # their R² scores.
#             # =================================================

#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "K-Neighbors Regressor": KNeighborsRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Random Forest": RandomForestRegressor(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#                 "Gradient Boosting Regressor": GradientBoostingRegressor()
#             }

#             # =================================================
#             # STEP 3: TRAIN AND EVALUATE ALL MODELS
#             # =================================================
#             #
#             # evaluate_models() will:
#             #
#             #     1. Train every model
#             #     2. Make predictions
#             #     3. Calculate R² score
#             #     4. Store the score
#             #
#             # =================================================

#             model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

#             # =================================================
#             # STEP 4: FIND THE BEST MODEL SCORE
#             # =================================================
#             #
#             # model_report.values() contains all R² scores.
#             #
#             # Example:
#             #
#             # [0.87, 0.91, 0.89, 0.85]
#             #
#             # max() returns:
#             #
#             # 0.91
#             #
#             # Therefore:
#             #
#             # best_model_score = 0.91
#             # =================================================

#             best_model_score = max(sorted(model_report.values()))

#             # =================================================
#             # STEP 5: FIND THE BEST MODEL NAME
#             # =================================================
#             #
#             # We now know the highest score.
#             #
#             # Next, we find which model produced that score.
#             #
#             # Example:
#             #
#             # Random Forest → 0.91
#             #
#             # Therefore:
#             #
#             # best_model_name = "Random Forest"
#             # =================================================

#             best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

#             # =================================================
#             # STEP 6: GET THE BEST MODEL OBJECT
#             # =================================================
#             #
#             # Get the actual model object from our dictionary.
#             #
#             # Example:
#             #
#             # best_model_name = "Random Forest"
#             #
#             # best_model =
#             # RandomForestRegressor(...)
#             #
#             # IMPORTANT:
#             #
#             # evaluate_models() already called model.fit()
#             # on every model.
#             #
#             # Therefore this model is already trained.
#             # =================================================

#             best_model = models[best_model_name]

#             # =================================================
#             # STEP 7: CHECK MINIMUM MODEL PERFORMANCE
#             # =================================================
#             #
#             # We only accept the best model if its R² score
#             # is at least 0.60.
#             #
#             # If:
#             #
#             # R² < 0.60
#             #
#             # we consider the model performance insufficient.
#             # =================================================

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found")

#             logging.info(f"Best found model on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")

#             # =================================================
#             # STEP 8: SAVE THE BEST MODEL
#             # =================================================
#             #
#             # Save the trained model as:
#             #
#             # artifacts/model.pkl
#             #
#             # This allows us to load the model later without
#             # training it again.
#             # =================================================

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             # =================================================
#             # STEP 9: MAKE FINAL PREDICTIONS
#             # =================================================
#             #
#             # The best model has already been trained inside
#             # evaluate_models().
#             #
#             # Now use it to predict the test data.
#             # =================================================

#             predicted = best_model.predict(X_test)

#             # =================================================
#             # STEP 10: CALCULATE FINAL R² SCORE
#             # =================================================
#             #
#             # Compare:
#             #
#             # y_test
#             #     ↓
#             # Actual math scores
#             #
#             # predicted
#             #     ↓
#             # Model's predicted math scores
#             #
#             # The result is the final R² score.
#             # =================================================

#             r2_square = r2_score(y_test, predicted)
            
#             # Return the final model performance
#             return r2_square

#         except Exception as e:
#             raise CustomException(e, sys)