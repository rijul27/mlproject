import os
import sys
import pickle

from src.exception import CustomException


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