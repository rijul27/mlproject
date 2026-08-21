import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

# ====================================================================================
# DATA INGESTION CONFIGURATION
# =====================================================================================
# This class stores all the paths required during the data ingestion process.
# Instead of writing the same paths everywhere in our code, we keep them in one place.
# ======================================================================================

@dataclass
class DataIngestionConfig:

    # Path where the training dataset will be saved
    train_data_path: str = os.path.join("artifacts","train.csv")

    # Path where the testing dataset will be saved
    test_data_path: str = os.path.join("artifacts","test.csv")

    # Path where the complete/raw dataset will be saved
    raw_data_path: str = os.path.join("artifacts","data.csv")


# ============================================================
# DATA INGESTION CLASS
# ============================================================

class DataIngestion:

    def __init__(self):

        # Create an object of DataIngestionConfig.
        # This gives us access to:
        # self.ingestion_config.train_data_path
        # self.ingestion_config.test_data_path
        # self.ingestion_config.raw_data_path

        self.ingestion_config = DataIngestionConfig()

    # ========================================================
    # DATA INGESTION METHOD
    # ========================================================

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")

        try:

            # ------------------------------------------------
            # STEP 1: READ THE DATASET
            # ------------------------------------------------
            # Read the original CSV file using pandas.
            # Your dataset is located at:
            # notebook/data/student_performance.csv
            # ------------------------------------------------

            df = pd.read_csv("notebook/data/stud.csv")

            logging.info("Read the dataset as dataframe")


            # ------------------------------------------------
            # STEP 2: CREATE ARTIFACTS DIRECTORY
            # ------------------------------------------------
            # We want to store the processed files inside:  artifacts/
            # exist_ok=True means that Python won't throw
            # an error if the directory already exists.
            # ------------------------------------------------

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # ------------------------------------------------
            # STEP 3: SAVE RAW DATA
            # ------------------------------------------------
            # Save the complete dataset before splitting it.
            # This gives us a copy of the original dataset
            # inside our ML pipeline.
            # ------------------------------------------------

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw data saved successfully")


            # ------------------------------------------------
            # STEP 4: TRAIN-TEST SPLIT
            # ------------------------------------------------
            # Split the dataset into:
            # 80% → Training data
            # 20% → Testing data
            # random_state=42 ensures that we get the same
            # split every time we run the code.
            # ------------------------------------------------

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)


            # ------------------------------------------------
            # STEP 5: SAVE TRAINING DATA
            # ------------------------------------------------

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            # ------------------------------------------------
            # STEP 6: SAVE TESTING DATA
            # ------------------------------------------------

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            # ------------------------------------------------
            # STEP 7: LOG SUCCESS
            # ------------------------------------------------

            logging.info("Ingestion of the data is completed")

            # ------------------------------------------------
            # STEP 8: RETURN TRAIN AND TEST PATHS
            # ------------------------------------------------
            # These paths can be used by the next component
            # of the ML pipeline.
            # ------------------------------------------------

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        # ====================================================
        # EXCEPTION HANDLING
        # ====================================================

        except Exception as e:

            # If anything goes wrong:
            #
            # 1. Capture the original error
            # 2. Send it to CustomException
            # 3. CustomException gives us:
            #    - file name
            #    - line number
            #    - error message

            raise CustomException(e, sys)


# =================================================================================================
# INITIAL IMPLEMENTATION — DATA INGESTION ONLY
# =================================================================================================
#
# PURPOSE:
# This was the initial implementation of the Data Ingestion component.
#
# At the beginning of the project, I tested Data Ingestion separately before
# connecting it with the complete Machine Learning pipeline.
#
# Initial flow:
#
#     Raw Dataset
#          ↓
#     Data Ingestion
#          ↓
#     train.csv + test.csv
#
# This approach was used only to understand and test the Data Ingestion
# component independently.
#
# Later, Data Ingestion was connected with:
#
#     Data Ingestion
#          ↓
#     Data Transformation
#          ↓
#     Model Training
#
# The complete pipeline is implemented in the MAIN MACHINE LEARNING PIPELINE
# section below.
# =================================================================================================

# if __name__ == "__main__":
#
#     # Create an object of DataIngestion
#     obj = DataIngestion()
#
#     # Start the Data Ingestion process
#     obj.initiate_data_ingestion()


# =================================================================================================
# SECOND IMPLEMENTATION — DATA INGESTION + DATA TRANSFORMATION
# =================================================================================================
#
# PURPOSE:
# After testing Data Ingestion independently, the next step was to connect
# Data Ingestion with the Data Transformation component.
#
# At this stage, the output of Data Ingestion became the input of
# Data Transformation.
#
# FLOW:
#
#     Raw Dataset
#          ↓
#     Data Ingestion
#          ↓
#     train.csv + test.csv
#          ↓
#     Data Transformation
#          ↓
#     Transformed train/test data
#          ↓
#     preprocessor.pkl
#
# WHAT WAS LEARNED IN THIS STEP:
#
#     1. How to pass the output of one component to another component.
#     2. Data Ingestion returns the paths of train.csv and test.csv.
#     3. These paths are passed to Data Transformation.
#     4. Data Transformation preprocesses the training and testing data.
#     5. The preprocessing object is saved as preprocessor.pkl.
#
# This was the second stage of building the complete ML pipeline.
# Later, Model Training was added as the next component.
# =================================================================================================


# if __name__ == "__main__":
#
#     # =============================================================================================
#     # STEP 1: DATA INGESTION
#     # =============================================================================================
#     #
#     # Create the DataIngestion object and start the Data Ingestion process.
#     #
#     # The component:
#     #     - Reads the raw dataset
#     #     - Creates the artifacts directory
#     #     - Saves the raw dataset
#     #     - Splits the data into train and test sets
#     #     - Saves train.csv and test.csv
#     #
#     # It returns the paths of train.csv and test.csv.
#     # =============================================================================================
#
#     obj = DataIngestion()
#
#
#     # Get the paths generated by Data Ingestion.
#
#     train_data_path, test_data_path = (
#         obj.initiate_data_ingestion()
#     )
#
#
#     # =============================================================================================
#     # STEP 2: DATA TRANSFORMATION
#     # =============================================================================================
#     #
#     # Create the DataTransformation object.
#     #
#     # The train.csv and test.csv paths generated by Data Ingestion
#     # are passed to Data Transformation.
#     #
#     # Data Transformation:
#     #     - Handles missing values
#     #     - Encodes categorical features
#     #     - Scales numerical features
#     #     - Transforms train and test data
#     #     - Saves the preprocessing object
#     #
#     # The preprocessing object is saved as:
#     #
#     #     artifacts/preprocessor.pkl
#     # =============================================================================================
#
#     data_transformation = DataTransformation()
#
#
#     data_transformation.initiate_data_transformation(
#         train_path=train_data_path,
#         test_path=test_data_path
#     )
# ====================================================================================================

# =================================================================================================
# FINAL IMPLEMENTATION — COMPLETE MACHINE LEARNING PIPELINE
# =================================================================================================
#
# PURPOSE:
# This is the final implementation of the complete Machine Learning pipeline.
#
# During the development of this project, the pipeline was built step-by-step:
#
#     1st Implementation → Data Ingestion only
#     2nd Implementation → Data Ingestion + Data Transformation
#     3rd Implementation → Complete ML Pipeline
#
# This final implementation connects all major components:
#
#     1. Data Ingestion
#     2. Data Transformation
#     3. Model Training
#
#
# COMPLETE PIPELINE FLOW:
#
#
#                         RAW DATASET
#                              │
#                              ▼
#                    ┌──────────────────┐
#                    │  DATA INGESTION  │
#                    └────────┬─────────┘
#                             │
#                    ┌────────┴────────┐
#                    ▼                 ▼
#               train.csv          test.csv
#                    │                 │
#                    └────────┬────────┘
#                             ▼
#                  ┌──────────────────────┐
#                  │ DATA TRANSFORMATION  │
#                  └──────────┬───────────┘
#                             │
#                    ┌────────┴────────┐
#                    ▼                 ▼
#                train_arr          test_arr
#                    │                 │
#                    └────────┬────────┘
#                             ▼
#                    ┌──────────────────┐
#                    │  MODEL TRAINING  │
#                    └────────┬─────────┘
#                             │
#                      Train Multiple
#                         Models
#                             │
#                             ▼
#                      Compare R² Scores
#                             │
#                             ▼
#                        Best Model
#                             │
#                             ▼
#                  artifacts/model.pkl
#
#
# ADDITIONAL ARTIFACT:
#
# Data Transformation also saves:
#
#     artifacts/preprocessor.pkl
#
#
# IMPORTANT:
#
# The output of one component becomes the input of the next component.
#
#     Data Ingestion
#           ↓
#     train.csv + test.csv
#           ↓
#     Data Transformation
#           ↓
#     train_arr + test_arr
#           ↓
#     Model Training
#           ↓
#     Best Model
#           ↓
#     model.pkl
#
# This creates a complete end-to-end Machine Learning pipeline.
#
# =================================================================================================


if __name__ == "__main__":

    # =============================================================================================
    # STEP 1: DATA INGESTION
    # =============================================================================================
    #
    # PURPOSE:
    # Read the original dataset and prepare the training and testing datasets.
    #
    # Data Ingestion:
    #
    #     - Reads the original dataset
    #     - Saves the raw dataset
    #     - Performs train-test split
    #     - Saves train.csv
    #     - Saves test.csv
    #
    # OUTPUT:
    #
    #     train_data_path
    #     test_data_path
    #
    # Example:
    #
    #     train_data_path = "artifacts/train.csv"
    #     test_data_path  = "artifacts/test.csv"
    #
    # =============================================================================================

    obj = DataIngestion()

    train_data_path, test_data_path = (
        obj.initiate_data_ingestion()
    )


    # =============================================================================================
    # STEP 2: DATA TRANSFORMATION
    # =============================================================================================
    #
    # PURPOSE:
    # Convert the train and test datasets into a format that can be used by
    # machine learning models.
    #
    # Data Transformation performs:
    #
    #     - Missing value handling
    #     - Numerical feature scaling
    #     - Categorical feature encoding
    #
    # It also saves:
    #
    #     artifacts/preprocessor.pkl
    #
    # INPUT:
    #
    #     train_data_path
    #     test_data_path
    #
    # OUTPUT:
    #
    #     train_arr
    #     test_arr
    #     preprocessor.pkl
    #
    # =============================================================================================

    data_transformation = DataTransformation()

    train_arr, test_arr, _ = (
        data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )
    )


    # =============================================================================================
    # STEP 3: MODEL TRAINING
    # =============================================================================================
    #
    # PURPOSE:
    # Train multiple machine learning regression models and select the best-performing model.
    #
    # Model Training:
    #
    #     1. Separates features (X) and target (y)
    #     2. Creates multiple regression models
    #     3. Trains every model
    #     4. Makes predictions on test data
    #     5. Calculates R² score for every model
    #     6. Compares the model scores
    #     7. Selects the model with the highest R² score
    #     8. Saves the best trained model
    #
    # OUTPUT:
    #
    #     artifacts/model.pkl
    #
    # The method also returns the final R² score of the selected model.
    #
    # =============================================================================================

    model_trainer = ModelTrainer()

    model_score = model_trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )


    # =============================================================================================
    # STEP 4: DISPLAY FINAL MODEL SCORE
    # =============================================================================================
    #
    # PURPOSE:
    # Display the final R² score returned by the ModelTrainer.
    #
    # Example:
    #
    #     Final Model R² Score: 0.89
    #
    # A higher R² score generally indicates better performance on the test dataset.
    #
    # =============================================================================================

    print(
        f"\nFinal Model R² Score: {model_score}"
    )