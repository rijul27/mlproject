import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# ============================================================
# 1. DATA TRANSFORMATION CONFIGURATION
# ============================================================
# This class stores the paths/configuration required during
# the data transformation process.
#
# We will save the preprocessing object as:
#
# artifacts/preprocessor.pkl
#
# This object is important because the same preprocessing
# steps must be applied to new/unseen data during prediction.
# ============================================================

@dataclass
class DataTransformationConfig:

    # Path where the preprocessing object will be saved
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )

# ============================================================
# 2. DATA TRANSFORMATION CLASS
# ============================================================
class DataTransformation:

    def __init__(self):
        # Create an object of DataTransformationConfig.
        # This gives us access to:
        # self.data_transformation_config.preprocessor_obj_file_path
        self.data_transformation_config = DataTransformationConfig()

    # ========================================================
    # 3. CREATE PREPROCESSING OBJECT
    # ========================================================
    # This method creates the preprocessing pipelines for:
    #
    # 1. Numerical columns
    # 2. Categorical columns
    #
    # These pipelines are combined using ColumnTransformer.
    # ========================================================
    def get_data_transformer_object(self):
        try:
            # Define the numerical and categorical columns
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # =================================================
            # NUMERICAL PIPELINE
            # =================================================
            #
            # Step 1:
            # SimpleImputer(strategy="median")
            #
            # If a numerical value is missing, replace it
            # with the median of that column.
            #
            # Step 2:
            # StandardScaler()
            # Standardizes numerical features.
            #
            # Formula: z = (x - mean) / standard deviation
            #
            # This makes numerical features easier for many
            # machine learning algorithms to work with.
            # =================================================
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            # =================================================
            # CATEGORICAL PIPELINE
            # =================================================
            #
            # Step 1:
            # SimpleImputer(strategy="most_frequent")
            #
            # Missing categorical values are replaced with
            # the most frequently occurring category.
            #
            # Step 2:
            # OneHotEncoder()  Converts categories into numerical columns.
            #
            # Step 3:
            # StandardScaler(with_mean=False)
            #
            # Scales the encoded categorical features.
            #
            # with_mean=False is used because OneHotEncoder
            # normally produces a sparse matrix.
            # =================================================

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns Scaling Completed: {numerical_columns}")
            logging.info(f"Categorical columns Encoding Completed: {categorical_columns}")

            # =================================================
            # COLUMN TRANSFORMER
            # =================================================
            #
            # ColumnTransformer allows us to apply different
            # preprocessing pipelines to different columns.
            #
            # Numerical columns
            #       ↓
            # num_pipeline
            #
            # Categorical columns
            #       ↓
            # cat_pipeline
            #
            # Finally, both transformed outputs are combined
            # into one feature matrix.
            # =================================================
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            # Return the complete preprocessing object
            return preprocessor

        except Exception as e:
            # If anything goes wrong, raise our custom exception
            raise CustomException(e, sys)

    # ========================================================
    # 4. INITIATE DATA TRANSFORMATION
    # ========================================================
    #
    # This method:
    # 1. Reads train.csv and test.csv
    # 2. Separates features and target
    # 3. Creates preprocessing object
    # 4. Fits preprocessing on training data
    # 5. Transforms training and testing data
    # 6. Combines transformed features with target
    # 7. Saves preprocessing object
    # 8. Returns transformed data
    # ========================================================
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # READ TRAINING AND TESTING DATA
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            # CREATE PREPROCESSING OBJECT
            logging.info("Obtaining the preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            
            # Separate the target column from the features
            target_column_name = "math_score"
            # Separate the input features and target variable for both training and testing datasets
            numerical_columns = ["reading_score", "writing_score"]

            # Separate input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # APPLY PREPROCESSING
            #
            # IMPORTANT:
            #
            # fit_transform() is used ONLY on training data.
            #
            # Why?
            #
            # The preprocessing object learns things such as:
            #
            # - Median values
            # - Most frequent categories
            # - Mean
            # - Standard deviation
            # - Categories for OneHotEncoder
            #
            # from the training data.
            #
            # Then we use transform() on test data.
            #
            # This prevents data leakage.
            # Fit and transform the training data, transform the testing data
            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Preprocessing completed successfully.")


            # COMBINE TRANSFORMED FEATURES AND TARGET
            #
            # After preprocessing, the input features (X) and target (y)
            # are still stored separately.
            #
            # input_feature_train_arr → transformed training features (X)
            # target_feature_train_df → training target (y = math_score)
            #
            # We combine them column-wise so that we have one complete
            # training array containing:
            #
            #     [transformed features | target]
            #
            # Example:
            #
            # X (transformed)              y
            # [1, 0, 0.25, -0.14]         61
            # [0, 1, 0.82,  0.91]          70
            #
            # After combining:
            #
            # [1, 0, 0.25, -0.14, 61]
            # [0, 1, 0.82,  0.91, 70]
            #
            # np.c_[] concatenates the arrays column-wise.
            #
            # IMPORTANT:
            # Combining X and y is NOT required by the ML model.
            # It is done here so that the Data Transformation component
            # can return the complete transformed dataset as one array
            # to the next component of the ML pipeline.
            # ============================================================
            # Combine the transformed features with the target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            # STEP 8: SAVE PREPROCESSOR
            #
            # We save the preprocessing object as:
            #
            # artifacts/preprocessor.pkl
            #
            # Why?
            #
            # During prediction, new data must go through
            # exactly the same preprocessing steps.
            #
            # We don't want to create a new scaler/encoder.
            # We reuse the one learned from training data.

            # Save the preprocessor object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            # STEP 9: RETURN RESULTS
            #
            # Return:
            #
            # 1. Transformed training data
            # 2. Transformed testing data
            # 3. Location of saved preprocessing object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            # Convert any error into our custom exception
            raise CustomException(e, sys)
        
