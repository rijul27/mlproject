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
    #
    # This method is responsible for:
    #
    # 1. Separating X and y
    # 2. Creating multiple ML models
    # 3. Training and evaluating all models
    # 4. Finding the best model
    # 5. Checking whether the model is good enough
    # 6. Saving the best model
    # 7. Returning the final R² score
    # ========================================================

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # =================================================
            # STEP 1: SEPARATE FEATURES AND TARGET
            # =================================================
            #
            # Data Transformation created arrays like:
            #
            # [feature1, feature2, feature3, ..., target]
            #
            # The LAST column is our target: math_score
            #
            # Therefore:
            #
            # [:, :-1]
            # → all rows, all columns except last
            # → X
            #
            # [:, -1]
            # → all rows, only the last column
            # → y
            # =================================================

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

            # =================================================
            # STEP 2: CREATE MULTIPLE ML MODELS
            # =================================================
            #
            # We don't know beforehand which algorithm will
            # perform best on our dataset.
            #
            # Therefore, we train several models and compare
            # their R² scores.
            # =================================================

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

            # =================================================
            # STEP 3: TRAIN AND EVALUATE ALL MODELS
            # =================================================
            #
            # evaluate_models() will:
            #
            #     1. Train every model
            #     2. Make predictions
            #     3. Calculate R² score
            #     4. Store the score
            #
            # =================================================

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # =================================================
            # STEP 4: FIND THE BEST MODEL SCORE
            # =================================================
            #
            # model_report.values() contains all R² scores.
            #
            # Example:
            #
            # [0.87, 0.91, 0.89, 0.85]
            #
            # max() returns:
            #
            # 0.91
            #
            # Therefore:
            #
            # best_model_score = 0.91
            # =================================================

            best_model_score = max(sorted(model_report.values()))

            # =================================================
            # STEP 5: FIND THE BEST MODEL NAME
            # =================================================
            #
            # We now know the highest score.
            #
            # Next, we find which model produced that score.
            #
            # Example:
            #
            # Random Forest → 0.91
            #
            # Therefore:
            #
            # best_model_name = "Random Forest"
            # =================================================

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # =================================================
            # STEP 6: GET THE BEST MODEL OBJECT
            # =================================================
            #
            # Get the actual model object from our dictionary.
            #
            # Example:
            #
            # best_model_name = "Random Forest"
            #
            # best_model =
            # RandomForestRegressor(...)
            #
            # IMPORTANT:
            #
            # evaluate_models() already called model.fit()
            # on every model.
            #
            # Therefore this model is already trained.
            # =================================================

            best_model = models[best_model_name]

            # =================================================
            # STEP 7: CHECK MINIMUM MODEL PERFORMANCE
            # =================================================
            #
            # We only accept the best model if its R² score
            # is at least 0.60.
            #
            # If:
            #
            # R² < 0.60
            #
            # we consider the model performance insufficient.
            # =================================================

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")

            # =================================================
            # STEP 8: SAVE THE BEST MODEL
            # =================================================
            #
            # Save the trained model as:
            #
            # artifacts/model.pkl
            #
            # This allows us to load the model later without
            # training it again.
            # =================================================

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # =================================================
            # STEP 9: MAKE FINAL PREDICTIONS
            # =================================================
            #
            # The best model has already been trained inside
            # evaluate_models().
            #
            # Now use it to predict the test data.
            # =================================================

            predicted = best_model.predict(X_test)

            # =================================================
            # STEP 10: CALCULATE FINAL R² SCORE
            # =================================================
            #
            # Compare:
            #
            # y_test
            #     ↓
            # Actual math scores
            #
            # predicted
            #     ↓
            # Model's predicted math scores
            #
            # The result is the final R² score.
            # =================================================

            r2_square = r2_score(y_test, predicted)
            
            # Return the final model performance
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

