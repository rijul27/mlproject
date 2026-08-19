import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


# ====================================================================================
# DATA INGESTION CONFIGURATION
# =====================================================================================
# This class stores all the paths required during the data ingestion process.
# Instead of writing the same paths everywhere in our code, we keep them in one place.
# ======================================================================================

@dataclass
class DataIngestionConfig:

    # Path where the training dataset will be saved
    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    # Path where the testing dataset will be saved
    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    # Path where the complete/raw dataset will be saved
    raw_data_path: str = os.path.join(
        "artifacts",
        "data.csv"
    )


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

        logging.info(
            "Entered the data ingestion method or component"
        )

        try:

            # ------------------------------------------------
            # STEP 1: READ THE DATASET
            # ------------------------------------------------
            # Read the original CSV file using pandas.
            # Your dataset is located at:
            # notebook/data/student_performance.csv
            # ------------------------------------------------

            df = pd.read_csv(
                "notebook/data/stud.csv"
            )

            logging.info(
                "Read the dataset as dataframe"
            )


            # ------------------------------------------------
            # STEP 2: CREATE ARTIFACTS DIRECTORY
            # ------------------------------------------------
            # We want to store the processed files inside:  artifacts/
            # exist_ok=True means that Python won't throw
            # an error if the directory already exists.
            # ------------------------------------------------

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )


            # ------------------------------------------------
            # STEP 3: SAVE RAW DATA
            # ------------------------------------------------
            # Save the complete dataset before splitting it.
            # This gives us a copy of the original dataset
            # inside our ML pipeline.
            # ------------------------------------------------

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Raw data saved successfully"
            )


            # ------------------------------------------------
            # STEP 4: TRAIN-TEST SPLIT
            # ------------------------------------------------
            # Split the dataset into:
            # 80% → Training data
            # 20% → Testing data
            # random_state=42 ensures that we get the same
            # split every time we run the code.
            # ------------------------------------------------

            logging.info(
                "Train test split initiated"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )


            # ------------------------------------------------
            # STEP 5: SAVE TRAINING DATA
            # ------------------------------------------------

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )


            # ------------------------------------------------
            # STEP 6: SAVE TESTING DATA
            # ------------------------------------------------

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )


            # ------------------------------------------------
            # STEP 7: LOG SUCCESS
            # ------------------------------------------------

            logging.info(
                "Ingestion of the data is completed"
            )


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


# ============================================================
# MAIN
# ============================================================
# This code runs only when this file is executed directly.
#
# Example: python src/components/data_ingestion.py
# ============================================================

if __name__ == "__main__":

    # Create DataIngestion object
    obj = DataIngestion()

    # Start the data ingestion process
    obj.initiate_data_ingestion()
