import sys

# Import the logging configuration from logger.py.
#
# logger.py creates the logs folder, creates the log file,
# and configures Python's logging system.
from src.logger import logging


# ============================================================
# 1. FUNCTION TO CREATE DETAILED ERROR MESSAGE
# ============================================================

def error_message_detail(error, error_detail: sys):
    """
    Creates a detailed error message.

    The function finds:

    1. Python file where the error occurred
    2. Line number where the error occurred
    3. Actual error message

    Parameters
    ----------
    error:
        The original exception/error.

    error_detail:
        The sys module, which provides information about
        the current exception.
    """

    # --------------------------------------------------------
    # Get exception information
    # --------------------------------------------------------
    #
    # sys.exc_info() returns three values:
    #
    # 1. Exception type
    # 2. Exception value
    # 3. Traceback information
    #
    # We only need the traceback, so we use "_"
    # for the first two values.
    # --------------------------------------------------------

    _, _, exc_tb = error_detail.exc_info()


    # --------------------------------------------------------
    # Get the Python file name where the error occurred
    # --------------------------------------------------------
    #
    # exc_tb
    #   ↓
    # traceback
    #
    # tb_frame
    #   ↓
    # current execution frame
    #
    # f_code
    #   ↓
    # code information
    #
    # co_filename
    #   ↓
    # Python file name
    # --------------------------------------------------------

    file_name = exc_tb.tb_frame.f_code.co_filename


    # --------------------------------------------------------
    # Create a detailed error message
    # --------------------------------------------------------

    error_message = (
        "Error occurred in Python script name [{0}] "
        "line number [{1}] "
        "error message [{2}]"
    ).format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )


    # Return the detailed error message
    return error_message


# ============================================================
# 2. CREATE CUSTOM EXCEPTION CLASS
# ============================================================

class CustomException(Exception):
    """
    Custom exception class.

    This class extends Python's built-in Exception class.

    The purpose of this class is to provide more useful
    information when an error occurs.
    """

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------
    #
    # __init__() runs automatically when we create an object.
    #
    # Example:
    #
    # CustomException(error, sys)
    #
    # --------------------------------------------------------

    def __init__(self, error_message, error_detail: sys):

        # Call the constructor of the parent Exception class.
        super().__init__(error_message)


        # ----------------------------------------------------
        # Create our detailed error message
        # ----------------------------------------------------

        self.error_message = error_message_detail(
            error_message,
            error_detail=error_detail
        )


    # --------------------------------------------------------
    # __str__()
    # --------------------------------------------------------
    #
    # This method controls what is displayed when we do:
    #
    # print(exception)
    #
    # or when Python displays the exception.
    # --------------------------------------------------------

    def __str__(self):

        return self.error_message


# ============================================================
# 3. TEST THE CUSTOM EXCEPTION
# ============================================================
#
# This block executes only when we run:
#
# python src/exception.py
#
# It will NOT execute when exception.py is imported.
# ============================================================

# if __name__ == "__main__":

#     try:

#         # ----------------------------------------------------
#         # Intentionally create an error.
#         #
#         # Dividing a number by zero causes:
#         #
#         # ZeroDivisionError
#         # ----------------------------------------------------

#         a = 1 / 0


#     except Exception as e:

#         # ----------------------------------------------------
#         # Write information about the error to the log file.
#         # ----------------------------------------------------

#         logging.info("Divide by Zero Error")


#         # ----------------------------------------------------
#         # Raise our custom exception.
#         #
#         # e   → original error
#         # sys → provides traceback information
#         # ----------------------------------------------------

#         raise CustomException(e, sys)




