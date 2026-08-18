import logging
import os
from datetime import datetime


# ============================================================
# 1. CREATE A LOG FILE NAME
# ============================================================
# datetime.now() gets the current date and time.
#
# strftime() converts the date/time into a readable format.
#
# Example:
# 2026-08-18_12-30-45.log
#
# Every time the application starts, a new log file
# will be created with a different timestamp.
# ============================================================

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"


# ============================================================
# 2. GET THE CURRENT WORKING DIRECTORY
# ============================================================
# os.getcwd() returns the directory from which the
# Python program is being executed.
#
# Example:
# D:\mlproject
# ============================================================

current_directory = os.getcwd()


# ============================================================
# 3. CREATE THE LOGS FOLDER PATH
# ============================================================
# We want our log files to be stored inside:
#
# D:\mlproject\logs
#
# os.path.join() creates the path correctly for Windows/Linux.
# ============================================================

logs_path = os.path.join(current_directory, "logs")


# ============================================================
# 4. CREATE THE LOGS DIRECTORY
# ============================================================
# If the "logs" folder does not exist, create it.
#
# exist_ok=True means:
#
# - If folder does not exist → create it
# - If folder already exists → don't throw an error
# ============================================================

os.makedirs(logs_path, exist_ok=True)


# ============================================================
# 5. CREATE THE COMPLETE LOG FILE PATH
# ============================================================
# Combine:
#
# logs folder + log file name
#
# Example:
# D:\mlproject\logs\2026-08-18_12-30-45.log
# ============================================================

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# ============================================================
# 6. CONFIGURE PYTHON LOGGING
# ============================================================
# basicConfig() tells Python how logging should work.
# ============================================================

logging.basicConfig(

    # Location where log messages will be stored
    filename=LOG_FILE_PATH,

    # Format of every log message
    #
    # %(asctime)s   → Date and time
    # %(lineno)d    → Line number
    # %(name)s      → Name of the logger/module
    # %(levelname)s → INFO / WARNING / ERROR / CRITICAL
    # %(message)s   → Actual message
    #
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # INFO is the minimum logging level we want to record.
    #
    # It will record:
    # INFO
    # WARNING
    # ERROR
    # CRITICAL
    #
    # DEBUG messages will not be recorded.
    level=logging.INFO
)


# ============================================================
# 7. TEST THE LOGGER
# ============================================================
# This code runs only when we execute logger.py directly.
#
# It will NOT run when logger.py is imported by another file.
# ============================================================

# if __name__ == "__main__":

#     logging.info("Logging has started")

#     print("Logging setup completed successfully!")
#     print(f"Log file created at: {LOG_FILE_PATH}")