# Student Performance Indicator — End-to-End Machine Learning Project

An end-to-end machine learning project that predicts a student's math score
based on demographic and academic factors — built and documented as a
complete pipeline: data ingestion → data transformation → model training
(with hyperparameter tuning) → a Flask prediction web app → cloud
deployment.

This document is both a project explanation and a beginner-friendly build
log — it walks through *what* was built, *why* each piece exists, and the
real errors hit (and fixed) along the way, since those are usually more
useful to learn from than a clean final version alone.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Project Overview](#project-overview)
3. [Tech Stack](#tech-stack)
4. [Project Architecture](#project-architecture)
5. [Environment & Project Setup](#environment--project-setup)
6. [Logging and Exception Handling](#logging-and-exception-handling)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Data Ingestion](#data-ingestion)
9. [Data Transformation](#data-transformation)
10. [Model Training](#model-training)
11. [Hyperparameter Tuning](#hyperparameter-tuning)
12. [Prediction Pipeline & Web App](#prediction-pipeline--web-app)
13. [Deployment](#deployment)
14. [How to Run This Project Locally](#how-to-run-this-project-locally)
15. [Common Errors & How They Were Fixed](#common-errors--how-they-were-fixed)
16. [Project Structure](#project-structure)
17. [Future Improvements](#future-improvements)

---

## Problem Statement

**Goal:** understand how a student's test performance is affected by other
variables, and build a model that predicts a student's **math score** from
those variables.

**Input features:**

| Feature | Type | Description |
|---|---|---|
| `gender` | categorical | Student's gender (2 values) |
| `race_ethnicity` | categorical | Ethnic group (5 groups, labeled generically) |
| `parental_level_of_education` | categorical | Highest education level of a parent (6 values) |
| `lunch` | categorical | Type of lunch the student receives (standard / free-reduced) |
| `test_preparation_course` | categorical | Whether the student completed a test-prep course (none / completed) |
| `reading_score` | numerical | Reading test score (0–100) |
| `writing_score` | numerical | Writing test score (0–100) |

**Target variable:** `math_score` (numerical, 0–100) — the value the model
is trained to predict.

**Why this dataset:** it's small (1,000 rows, 8 columns), has no missing
values, and mixes categorical and numerical features cleanly — which makes
it a good dataset for learning the *full pipeline* (EDA → preprocessing →
training → deployment) without the data itself being the hard part. The
same pipeline structure built here generalizes to messier, larger datasets.

**Type of problem:** this is a **regression** problem — the target
(`math_score`) is a continuous number, not a category, so the model is
evaluated using regression metrics (R², MAE, RMSE — explained in the
[Model Training](#model-training) section) rather than classification
metrics like accuracy or a confusion matrix.

---

## Project Overview

This project is built the way a machine learning solution is expected to
look in a real engineering setting — not just a single Jupyter notebook,
but a **modular, reusable pipeline** with proper logging, custom exception
handling, and a clean separation between:

- **Training code** (`src/components/`) — reads data, cleans/transforms it,
  trains and compares several models, and saves the best one.
- **Prediction code** (`src/pipeline/`) — loads the already-trained model
  and serves predictions for new input, without retraining anything.
- **A small web app** (`app.py` / `application.py`) — a Flask front end
  where a user fills in a form and gets a predicted math score back.
- **Three independent deployment paths** — AWS Elastic Beanstalk, a
  Dockerized deployment on AWS EC2/ECR, and a Dockerized deployment on
  Azure — covering both a fully-managed and a self-managed hosting style.

The project intentionally starts simple (a notebook, basic scaffolding) and
builds up in layers — each stage below documents one of those layers, in
the order they were actually built, including the mistakes made and fixed
along the way, because debugging real errors is a large part of what makes
a pipeline like this actually understandable.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8 |
| Data handling | pandas, numpy |
| ML / preprocessing | scikit-learn, CatBoost, XGBoost |
| Visualization (EDA) | matplotlib, seaborn |
| Web framework | Flask |
| Packaging | setuptools (`setup.py`), pip |
| Serialization | dill |
| Version control | Git + GitHub |
| Containerization | Docker |
| CI/CD | GitHub Actions, AWS CodePipeline, Azure Deployment Center |
| Cloud (AWS) | Elastic Beanstalk, EC2, ECR, IAM |
| Cloud (Azure) | Azure Container Registry (ACR), Web App for Containers |


## Project Architecture

At a high level, data flows through the project like this:

```
Raw CSV data
     │
     ▼
Data Ingestion  ──────►  splits into train.csv / test.csv, saves raw copy
     │
     ▼
Data Transformation  ──►  imputes missing values, encodes categoricals,
     │                    scales numeric features, saves preprocessor.pkl
     ▼
Model Training  ───────►  trains multiple regressors, tunes hyperparameters,
     │                    saves the best model as model.pkl
     ▼
Prediction Pipeline  ──►  loads preprocessor.pkl + model.pkl, transforms
     │                    new input, returns a prediction
     ▼
Flask Web App  ─────────►  a form where a user enters student details and
                            gets back a predicted math score
```

Everything to the left of the Flask app runs **once**, offline, to produce
two artifacts (`preprocessor.pkl` and `model.pkl`). The web app never
retrains anything — it only loads those two files and uses them.

---

## Environment & Project Setup

### Creating an isolated environment

The project uses a Python virtual environment created *inside* the project
folder (rather than in a central location), so every installed package
stays scoped to this project:

```bash
conda create -p venv python==3.8 -y
conda activate venv/
```

(Python 3.8 was used to match the original build environment — a newer
supported version such as 3.10+ works just as well for a fresh setup, since
3.8 has reached end-of-life.)

Without Anaconda, the same idea works with Python's built-in tool:
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### Git and `.gitignore`

```bash
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin <repo-url>
git push -u origin main
```

Key entries kept in `.gitignore` throughout the project (build artifacts
and generated files that should never be committed):
```
venv/
.venv/
__pycache__/
*.egg-info/
.env
artifacts/
```

### Why `artifacts/` is not committed

The `artifacts/` directory contains **generated ML files**, not source code:

```text
artifacts/
├── data.csv            # generated raw-data copy
├── train.csv           # generated training split
├── test.csv            # generated testing split
├── preprocessor.pkl    # fitted preprocessing pipeline
└── model.pkl           # trained/tuned ML model
```

These files are intentionally excluded from GitHub because they are generated
by the training pipeline and can be recreated from the source code and dataset.
This keeps the repository clean and makes it easier for a recruiter to focus on
the actual ML engineering work.

The repository contains everything required to recreate these artifacts:

```text
Data Ingestion
      ↓
Data Transformation
      ↓
Hyperparameter Tuning
      ↓
Best Model + Preprocessor
      ↓
artifacts/model.pkl
artifacts/preprocessor.pkl
```

After cloning the repository, run:

```bash
python src/components/data_ingestion.py
```

The training pipeline recreates the required files under `artifacts/`.

> **Recruiter note:** The project is intentionally structured as a reproducible
> end-to-end ML pipeline rather than depending on committed binary model files.
> The EDA notebooks, training code, hyperparameter-tuning logic, prediction
> pipeline, Flask application, and deployment configuration are included so the
> complete workflow can be reviewed and reproduced.


### `setup.py` — packaging the project

`setup.py` is what turns this project into an installable Python package
(the same mechanism behind `pip install <any-library>`), which is what lets
`src/` be imported as a package (`from src.logger import logging`, etc.)
throughout the rest of the codebase.

```python
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Read requirements.txt and return a clean list of package names."""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
```

`find_packages()` scans the project for any folder containing an
`__init__.py` file and treats it as an importable package — this is why
every folder under `src/` (`components/`, `pipeline/`) has its own empty
`__init__.py`: it's a marker, not functional code, but without it those
folders wouldn't be importable at all.

### `requirements.txt`

```
pandas
numpy
scikit-learn
seaborn
matplotlib
catboost
xgboost
dill
Flask
-e .
```

The final line, `-e .`, tells `pip` to install the project itself as an
**editable package** by running `setup.py` — this is why running
`pip install -r requirements.txt` also builds the project's own package
(visible as a `mlproject.egg-info/` folder appearing afterward, which is a
normal, gitignored build artifact).

```bash
pip install -r requirements.txt
```


## Logging and Exception Handling

Before writing any real pipeline logic, two cross-cutting pieces were built
first, since every other component in this project depends on them:
a **logger** (so anything that happens is written to a file automatically)
and a **custom exception** class (so errors report exactly which file and
line caused them, instead of a generic Python traceback).

### `src/logger.py`

```python
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```

**What this does, for anyone new to Python's `logging` module:**
- A new, timestamped log file is created every time the project runs
  (`logs/08_18_2026_14_32_05.log`, for example).
- `os.makedirs(logs_dir, exist_ok=True)` creates the `logs/` folder if it
  doesn't already exist — `exist_ok=True` prevents a crash on the second run.
- `logging.basicConfig(...)` is a one-time setup call. After this runs,
  anywhere in the project can call `logging.info("some message")` and it
  will be written to this file automatically, in the format specified —
  timestamp, line number, logger name, severity level, and the message
  itself.

**Important usage detail:** anywhere else in the project that needs to log
something must import logging **through** this module:
```python
from src.logger import logging
```
not a bare `import logging`. A bare import gives Python's default,
unconfigured logger (which prints to the console, not to a file) — the
`logging.basicConfig(...)` call above only takes effect once `src/logger.py`
has actually been imported and run.

### `src/exception.py`

```python
import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
```

**How it works:** `sys.exc_info()` returns `(exception_type, exception_value,
traceback_object)` — only the third value is used here. From the traceback
object, `.tb_frame.f_code.co_filename` gives the file where the error
occurred, and `.tb_lineno` gives the line number. `CustomException`
inherits from Python's built-in `Exception`, calls `super().__init__(...)`
to stay compatible with normal exception behavior, and overrides `__str__`
so printing the exception shows the detailed message rather than Python's
default representation.

**Typical usage, anywhere in the project:**
```python
import sys
from src.logger import logging
from src.exception import CustomException

try:
    a = 1 / 0
except Exception as e:
    logging.info("Divide by zero error")
    raise CustomException(e, sys)
```

Raising `CustomException(e, sys)` (both the caught error *and* the `sys`
module) produces a message like:
```
Error occurred in python script name [.../exception.py] line number [10]
error message [division by zero]
```

**A subtlety worth knowing:** `CustomException` relies on
`sys.exc_info()`, which only returns real data when Python is *currently
handling an active exception* — i.e., inside a `try/except` block. It's
designed to **wrap a caught error**, not to raise a message out of nowhere.


## Exploratory Data Analysis (EDA)

EDA was done in a Jupyter notebook (`notebook/1_EDA.ipynb`) rather than as a
`.py` script — EDA is inherently exploratory (running cells one at a time,
inspecting output, adjusting course based on what's found), which doesn't
fit a linear script well. The dataset itself lives at
`notebook/data/student.csv`.

### Basic checks

```python
df.head()
df.shape                # (1000, 8)
df.isnull().sum()       # missing values per column → all zero
df.duplicated().sum()   # duplicate rows → zero
df.info()               # column data types
df.nunique()             # unique values per column
df.describe()            # mean/std/min/max for numeric columns
```

**Findings:** no missing values and no duplicate rows in this dataset — so
no imputation or de-duplication was needed at the EDA stage itself (though
the transformation pipeline, described later, still includes an imputer as
a defensive default for any future data that *does* have gaps).

### Separating numeric vs. categorical columns

```python
numeric_features = [f for f in df.columns if df[f].dtype != 'O']
categorical_features = [f for f in df.columns if df[f].dtype == 'O']
```
`'O'` is pandas' dtype code for "object" (text/string) columns — anything
that isn't `'O'` is treated as numeric.

- **3 numeric features:** `math_score`, `reading_score`, `writing_score`
- **5 categorical features:** `gender`, `race_ethnicity`,
  `parental_level_of_education`, `lunch`, `test_preparation_course`

### Feature engineering (used for exploration, not part of the final model)

```python
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total_score'] / 3
```
These two derived columns were used to explore patterns (e.g. average score
by gender) during EDA. The final trained model predicts `math_score`
directly from the other columns — `total_score` and `average` were not used
as model inputs.

### Key observations from EDA

- Female students had a higher average score than male students across
  math, reading, and writing.
- A handful of students scored a perfect 100 in math and in writing;
  a similarly small group scored very low (≤20) in reading.
- Every categorical column has a small number of categories (2–6), which
  made one-hot encoding a reasonable choice for the preprocessing pipeline
  (rather than something like target encoding, which tends to matter more
  when a categorical column has dozens or hundreds of distinct values).

### A first model, trained in the notebook

Before writing any pipeline code, a first model was trained directly in the
notebook to validate the overall approach:

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

X = df.drop(columns=['math_score'], axis=1)
y = df['math_score']

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", OneHotEncoder(), cat_features),
        ("StandardScaler", StandardScaler(), num_features),
    ]
)

X = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

One-hot encoding expanded the 7 input columns to **19 columns** (since each
category in every categorical feature becomes its own 0/1 column).

Several regression models were trained and compared by **R² score**
(explained fully in [Model Training](#model-training)):

| Model | Approx. R² |
|---|---|
| Ridge | ~0.88 |
| Linear Regression | ~0.88 |
| CatBoosting Regressor | ~0.85 |
| *(others tried: Lasso, KNN, Decision Tree, Random Forest, AdaBoost, SVR, XGBoost)* | — |

Since Ridge and Linear Regression performed essentially the same, **Linear
Regression** was chosen for its simplicity — a reasonable default rule of
thumb: prefer the simplest model that performs comparably to more complex
ones.

This notebook result validated the approach before any of it was converted
into the modular, reusable code described in the following sections.


## Data Ingestion

**File:** `src/components/data_ingestion.py`

**Responsibility:** read the raw dataset from its source and split it into
train/test sets, saving all three (raw, train, test) as CSVs for the next
stage to consume. In a larger production setup, this is the one place that
would change if the data source were a database or an API instead of a
local CSV — everything downstream stays the same.

```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/student.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
```

**Why a `@dataclass` for the config:** `DataIngestionConfig` only holds
settings (three file paths), with no behavior — `@dataclass` auto-generates
the `__init__` that would otherwise need to be written by hand, just from
the class-level variable declarations. Keeping every input this component
needs in one small class (rather than scattered through the method body)
makes it easy to change later — e.g. pointing outputs somewhere else — in
one place.

**Key implementation notes:**
- `os.makedirs(os.path.dirname(path), exist_ok=True)` creates the
  `artifacts/` folder before anything tries to write into it;
  `exist_ok=True` avoids crashing on a second run.
- `to_csv(..., index=False, header=True)` — `index=False` skips writing
  pandas' internal row-numbering column into the file (almost always
  wanted); `header=True` keeps the column names as the first row.
- The method returns **file paths**, not the data itself — this keeps
  ingestion and the next stage (transformation) independent of each other;
  each can be tested on its own.
- Every step is logged, and the whole method is wrapped in
  `try/except → CustomException`, consistent with every other component in
  this project.

**Run it directly** (must be run from the project root — the path
`'notebook/data/student.csv'` is relative to the current working
directory):
```bash
python src/components/data_ingestion.py
```

**Output:**
```
artifacts/
├── data.csv     # full raw dataset, unmodified
├── train.csv    # 80% split
└── test.csv     # 20% split
```


## Data Transformation

**File:** `src/components/data_transformation.py`

**Responsibility:** turn the raw train/test CSVs into fully numeric arrays a
model can train on — impute missing values, encode categorical columns,
scale numeric columns — and save the fitted transformer so the exact same
transformation can be reapplied later to brand-new data at prediction time.

```python
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation based on the different types of data
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
```

**Understanding the pipeline design:**
- **`Pipeline(steps=[...])`** chains transformation steps so they run *in
  order* — missing values are filled in before scaling happens, since you
  can't scale a column that still has gaps in it.
- **`SimpleImputer(strategy="median")`** for numeric columns — median is
  used rather than mean because EDA found outliers in some numeric columns,
  and median is far less sensitive to extreme values.
- **`SimpleImputer(strategy="most_frequent")`** for categorical columns —
  fills gaps with the most common category (mode), since "median" doesn't
  apply to non-numeric data.
- **`StandardScaler(with_mean=False)` in the categorical pipeline — a real
  gotcha worth knowing.** After `OneHotEncoder`, the output is a **sparse
  matrix** (a memory-efficient representation that doesn't explicitly store
  the many zero values one-hot encoding produces). Standard scaling
  normally *centers* data by subtracting the mean, but centering a sparse
  matrix would fill in all those zeros with non-zero values — scikit-learn
  raises `ValueError: Cannot center sparse matrix: pass with_mean=False` if
  this isn't set. This is the standard fix any time `StandardScaler`
  immediately follows `OneHotEncoder` in the same pipeline.
- **`ColumnTransformer([(name, pipeline, columns), ...])`** applies each
  pipeline only to its listed columns, then combines the results into one
  array.

**`fit_transform` vs. `transform` — an important distinction:**
- `fit_transform(input_feature_train_df)` **learns** the transformation
  parameters (which categories exist, what the mean/median/std values are)
  *from the training data*, and applies them in the same step.
- `transform(input_feature_test_df)` only **applies** parameters already
  learned from training data — it never relearns anything from the test
  set. Calling `fit_transform` on test data too would leak information
  about the test set's distribution into preprocessing, making evaluation
  metrics unreliably optimistic — a mistake worth avoiding deliberately.

**`np.c_[array1, array2]`** glues two arrays together as columns, combining
the transformed features with the target column into a single array. This
is why `data_transformation.py` puts the target column last — it makes
splitting features and target back apart in the next stage trivial with a
single slice, rather than needing to track column names between components.

**`src/utils.py` — `save_object()`:**
```python
import os
import sys
import dill

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
```
`dill` is used instead of Python's built-in `pickle` module because it
reliably serializes a wider range of Python objects. `"wb"` (write-binary)
mode is required since pickled data is binary, not text. This function is
deliberately generic (`file_path`, `obj`) so it's reused later to save the
trained model itself, not just the preprocessor.

**Output added to `artifacts/`:**
```
artifacts/
└── preprocessor.pkl
```


## Model Training

**File:** `src/components/model_trainer.py`

**Responsibility:** take the transformed train/test arrays, train several
regression models, evaluate each, pick the best-performing one, and save it.

### Regression metrics used

Since `math_score` is a continuous value, this is a **regression** problem,
evaluated with:

- **MAE** (mean absolute error) — average size of the prediction error, in
  the same units as the target (e.g. "predictions are off by ~4.5 points
  on average").
- **MSE** (mean squared error) — like MAE, but squares each error first,
  which penalizes large mistakes more heavily than small ones.
- **RMSE** (root mean squared error) — the square root of MSE, bringing the
  units back in line with the original target so it's directly comparable
  to MAE, while still weighting large errors more.
- **R²** (R-squared) — roughly a 0–1 score representing how much of the
  variation in the target the model explains; closer to 1 is better. This
  is the metric used to compare and select the best model.

*(For a classification problem instead, metrics like accuracy, precision,
recall, and a confusion matrix would be used instead of these.)*

### `src/utils.py` — `evaluate_models()`

```python
from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
```

For every candidate model: fit on training data, predict on both train and
test sets, score both with R², and store only the **test** score in the
returned report — since what matters when choosing a model is performance
on unseen data, not how well it memorized the training set.

### `src/components/model_trainer.py`

```python
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models,
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
```

**Key implementation notes:**
- **`train_array[:, :-1]` / `train_array[:, -1]`** — NumPy slicing that
  splits the combined array back into features and target: `[:, :-1]` means
  "every row, every column except the last," and `[:, -1]` means "every
  row, only the last column." This exactly undoes the `np.c_[...]`
  column-stacking done in the transformation stage — which is precisely why
  the target was placed last there.
- **`CatBoostRegressor(verbose=False)`** — without this, CatBoost prints
  hundreds of lines of training progress; `verbose=False` keeps the console
  output clean for an automated pipeline.
- **Finding the best model**: `max(model_report.values())` gets the
  highest R² score; `.index(...)` finds where that score sits in the values
  list, and the same position in the keys list gives the matching model
  name; `models[best_model_name]` looks up the actual fitted model object.
- **The 0.6 threshold**: if even the best model scores below 60% R², the
  code raises an exception rather than silently saving a weak model — a
  simple guardrail against shipping something clearly not good enough.

**Result:** with default hyperparameters, R² landed around **~87%**.

**Output added to `artifacts/`:**
```
artifacts/
└── model.pkl
```


## Hyperparameter Tuning

Rather than training every model with scikit-learn's default settings, the
model trainer searches over a range of settings for each model and keeps
the best-performing combination — using **`GridSearchCV`**.

**Parameter vs. hyperparameter, quickly:** a *parameter* is something a
model *learns* from data during training (e.g. linear regression's
coefficients). A *hyperparameter* is something chosen **before** training
starts, controlling how the model learns (e.g. how many trees a Random
Forest builds). Hyperparameter tuning searches for good hyperparameter
values instead of guessing.

### The parameter grid

```python
params = {
    "Decision Tree": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    },
    "Random Forest": {
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "Gradient Boosting": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "Linear Regression": {},
    "XGBRegressor": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
    "CatBoosting Regressor": {
        "depth": [6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "iterations": [30, 50, 100],
    },
    "AdaBoost Regressor": {
        "learning_rate": [0.1, 0.01, 0.5, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },
}
```
`"Linear Regression": {}` is intentionally empty — plain linear regression
doesn't have meaningful hyperparameters to search the way tree-based or
boosting models do, so `GridSearchCV` with an empty grid just fits it once
with its defaults.

> A cleaner alternative for a larger project: move this dictionary into a
> separate config file (e.g. YAML) loaded at runtime, so tuning ranges can
> be edited without touching the training code. Kept inline here to keep
> this stage simple.

### Updated `evaluate_models()` — now with `GridSearchCV`

```python
from sklearn.model_selection import GridSearchCV

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
```

**What's happening:**
- **`GridSearchCV(model, para, cv=3)`** exhaustively tries **every
  combination** in a model's parameter grid — e.g. Gradient Boosting's grid
  (4 learning rates × 6 subsample values × 6 estimator counts) means 144
  combinations tried automatically.
- **`cv=3`** means 3-fold cross-validation: the training data is split into
  3 chunks; for each combination, the model trains on 2 chunks and
  validates on the remaining 1, three times over (rotating which chunk is
  held out), and the scores are averaged — a more reliable estimate of how
  well each combination generalizes than a single split.
- **`gs.best_params_`** holds the best-found hyperparameter combination
  after the search completes.
- **`model.set_params(**gs.best_params_)`** applies those settings back
  onto the model, then it's re-fit on the full training data.
  (`GridSearchCV` also stores this exact retrained model as
  `gs.best_estimator_`, which could be used directly as a shortcut instead.)

**Note:** this stage takes noticeably longer to run than training with
defaults, since every model is now fit dozens or hundreds of times instead
of once. For this dataset, final R² came out to roughly the same **~87%**
— for a small, clean dataset like this, tuning didn't move the needle much,
but the pipeline now performs a proper search rather than relying purely on
defaults, which matters more on larger or messier data.


## Prediction Pipeline & Web App

With a trained model and preprocessor saved to disk, the last piece is a
way to actually **use** them: a prediction pipeline, and a small Flask app
around it.

### `src/utils.py` — `load_object()`

The mirror image of `save_object()` — reads a `.pkl` file back into memory:

```python
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
```

### `src/pipeline/predict_pipeline.py`

Two classes: `CustomData` packages a single form submission into the shape
the model expects, and `PredictPipeline` loads the saved files and runs the
prediction.

```python
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
```

**Why every dictionary value is wrapped in a list** (`"gender":
[self.gender]`): pandas' `DataFrame` constructor expects each column's
values as a list-like object, even for a single value — a `DataFrame` is
fundamentally a table of columns. Without the brackets, `pd.DataFrame(...)`
raises `ValueError: If using all scalar values, you must pass an index`.

**Why `preprocessor.transform(features)`, not `fit_transform`:** same
reasoning as during training — the preprocessor was already fit once on
training data; at prediction time it should only apply those already-learned
rules to the new row, never relearn them from a single incoming submission.

### `app.py` — the Flask app

```python
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')),
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
```

**Flask basics, for anyone new to it:**
- `Flask(__name__)` creates the web application object.
- `@app.route('/')` is a decorator: "when someone visits this URL, run the
  function below it." That function is called a **view function**.
- `render_template('index.html')` looks inside a folder named exactly
  `templates/` (a fixed convention Flask expects) and returns its content
  as the page shown to the browser.
- `methods=['GET', 'POST']` lets one route handle both a normal page visit
  (`GET`) and a form submission (`POST`) — `request.method` tells you which
  one just happened.
- `request.form.get('gender')` pulls a submitted form field's value by its
  HTML `name` attribute.

**A real bug worth flagging explicitly:** `request.form.get(...)` always
returns a **string**, even for fields representing numbers. Since
`reading_score` and `writing_score` need to reach the numeric preprocessing
pipeline, they must be explicitly cast:
```python
reading_score=float(request.form.get('reading_score')),
writing_score=float(request.form.get('writing_score')),
```
Skipping this doesn't necessarily crash immediately, but silently feeds
string values into a numeric pipeline — worth catching explicitly.

**Why `application = Flask(__name__)` then `app = application`:** this is a
convention for compatibility with hosting platforms (like AWS Elastic
Beanstalk) that specifically look for a variable named `application` — see
the [Deployment](#deployment) section. Locally, either name works fine on
its own.

**Why `results[0]`, not `results`:** `model.predict(...)` always returns an
**array** of predictions, even for a single input row (scikit-learn's
convention). `results[0]` extracts the single number.

**A security note:** `debug=True` is convenient during development (auto
reload, detailed error pages) but should be **removed or set to `False`
before any public deployment** — leaving it on can let a visitor execute
arbitrary code on the server.

### Templates

```
templates/
├── index.html   # simple landing page at '/'
└── home.html    # the prediction form, at '/predictdata'
```

`home.html`'s `<form>` posts back to the same route using Flask's
`url_for()` helper (referencing the view function's *Python name*, not the
raw URL path, so the reference doesn't break if the route ever changes):
```html
<form action="{{ url_for('predict_datapoint') }}" method="post">
```
and displays the result once available:
```html
<h2>The prediction is {{ results }}</h2>
```

### Running it locally

```bash
python app.py
```
- `http://127.0.0.1:5000/` → the home page
- `http://127.0.0.1:5000/predictdata` → the form; submitting it returns a
  predicted math score


## Deployment

This project was deployed **three different ways**, across two cloud
providers — partly to compare a fully-managed hosting style against a more
hands-on, containerized one, since both are common in real production
settings and worth being able to speak to.

### Option 1 — AWS Elastic Beanstalk + CodePipeline (no Docker)

The simplest path: plain Python code, no containerization.

**Two extra files needed:**

`.ebextensions/python.config`:
```yaml
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```
This tells Elastic Beanstalk where the app's entry point lives: "look
inside `application.py`, and find a variable called `application`" — which
is exactly why `app.py` was written with:
```python
application = Flask(__name__)
app = application
```

`application.py` — a copy of `app.py`, with `debug=True` removed (a real
security consideration once the app is publicly reachable):
```python
if __name__ == "__main__":
    app.run(host="0.0.0.0")
```

**Setup steps:**
1. AWS Console → **Elastic Beanstalk** → Create application → choose the
   **Python** platform → start from a **sample application**.
2. AWS Console → **CodePipeline** → Create pipeline:
   - Source provider: **GitHub (Version 1)**, connected to the repository
     and `main` branch.
   - Build provider: skipped (no build stage needed for this app).
   - Deploy provider: **AWS Elastic Beanstalk**, pointing at the
     application/environment created above.
3. Once created, every push to `main` automatically triggers a redeploy —
   this is the "Continuous Delivery" half of CI/CD (automated deployment);
   no automated testing/build step was added in this path, so it isn't CI
   in the full sense.

**A real gotcha:** having *both* `app.py` and `application.py` present at
once caused the first deployment to fail — the fix was deleting `app.py`
entirely, keeping only the one file matching `WSGIPath`.

### Option 2 — Docker + GitHub Actions + AWS ECR + EC2 (self-managed)

A more hands-on path: the app is containerized, and a custom GitHub Actions
workflow builds, pushes, and deploys it onto a self-managed EC2 server.

**`Dockerfile`:**
```dockerfile
FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN apt update -y

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
```

**High-level setup:**
1. **IAM user** created with only two scoped permissions (principle of
   least privilege): `AmazonEC2ContainerRegistryFullAccess` and
   `AmazonEC2FullAccess` — generates an access key pair used by the
   pipeline to authenticate to AWS.
2. **ECR repository** created (private) to store the Docker image.
3. **EC2 instance** launched (Ubuntu, t2.medium), with Docker installed on
   it manually via Docker's official install script, and the `ubuntu` user
   added to the `docker` group so commands don't need `sudo`.
4. **GitHub self-hosted runner** registered on that same EC2 instance — a
   self-hosted runner is needed here because the final deploy step needs to
   run directly *on* the EC2 machine (pulling and starting the new Docker
   image there), which a temporary GitHub-hosted runner has no access to do.
5. **GitHub Secrets** added: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_REGION`, `AWS_ECR_LOGIN_URI`, `ECR_REPOSITORY_NAME` (just the
   repository name — **not** the full URL, a real mistake made and fixed
   during setup, which had caused the image push step to fail).
6. **`.github/workflows/main.yaml`** — a three-job workflow:
   - `integration` (placeholder for linting/tests) → runs on a normal
     GitHub-hosted runner.
   - `build-and-push-ecr-image` → builds the Docker image and pushes it to
     ECR, also on a GitHub-hosted runner.
   - `Continuous-Deployment` → `runs-on: self-hosted` — pulls the new image
     and restarts the container **on the EC2 instance itself**.
7. **Security group**: port **8080** opened as a custom TCP inbound rule
   (the app's configured port for this path) — without this, the container
   runs fine but the request never reaches it, blocked at AWS's
   network-level firewall.

**Cleanup after testing** (four separate steps, since these AWS resources
aren't grouped together): remove the self-hosted runner, terminate the EC2
instance, delete the ECR repository, delete the IAM user.

### Option 3 — Docker + Azure Container Registry + Web App for Containers

A Docker-based deployment on Azure, closer in spirit to Option 1's managed
style than Option 2's self-managed EC2 approach.

| Concept | AWS equivalent | Azure |
|---|---|---|
| Private image registry | ECR | **ACR** (Azure Container Registry) |
| Managed deploy target | Elastic Beanstalk | **Web App for Containers** |

**Setup steps:**
1. **Azure Container Registry** created; admin user enabled to allow
   `docker login` with a username/password.
2. Docker image built and pushed **once, manually**, to bootstrap the
   registry with an initial image:
   ```bash
   docker build -t <registry-login-server>/student-performance:latest .
   docker login <registry-login-server>
   docker push <registry-login-server>/student-performance:latest
   ```
3. **Web App for Containers** created, publish mode set to **Docker
   Container**, pointed at the image just pushed to ACR.
4. **Deployment Center** → Continuous Deployment turned **On**, GitHub
   repository and branch selected. Unlike Option 2, **Azure automatically
   generates and commits its own GitHub Actions workflow file** — nothing
   is hand-written here. From that point on, every push to the tracked
   branch triggers Azure's own workflow (build → push to ACR → redeploy)
   on regular GitHub-hosted runners — **no self-hosted runner needed at
   all**, since "deployment" here means telling the already-running,
   Azure-managed Web App to swap in a new image, not directly managing a
   server.

**Cleanup:** because everything was placed in a single Azure **resource
group**, deleting that one resource group tears down the registry, the web
app, and everything else at once — notably simpler than Option 2's
four-step AWS cleanup.

### Comparing all three

| | Elastic Beanstalk | Docker + EC2 (AWS) | Docker + Web App (Azure) |
|---|---|---|---|
| Containerized | No | Yes | Yes |
| Deploy target | Managed | Self-managed EC2 | Managed |
| Pipeline definition | AWS CodePipeline (console) | Hand-written GitHub Actions | Auto-generated GitHub Actions |
| Runner needed | N/A | Self-hosted (own EC2) | None |
| Setup effort | Low | High | Medium |
| Cleanup | Simple | 4 separate deletions | 1 resource-group deletion |

Each represents a different point on the control-vs-convenience spectrum —
managed platforms (Elastic Beanstalk, Azure Web Apps) trade some control
for the provider handling more infrastructure, while the EC2 + self-hosted
runner path gives full control over the server at the cost of managing
everything about it yourself (OS updates, Docker installs, security
patches, and so on).


## How to Run This Project Locally

```bash
# 1. Clone the repository
git clone <repo-url>
cd mlproject

# 2. Create and activate a virtual environment
conda create -p venv python==3.10 -y
conda activate venv/
# or, without conda:
# python -m venv venv && source venv/bin/activate  (Linux/Mac)
# python -m venv venv && venv\Scripts\activate      (Windows)

# 3. Install dependencies (also builds the project as an editable package)
pip install -r requirements.txt

# 4. Run the full training pipeline (ingestion → transformation → training)
python src/components/data_ingestion.py

# 5. Start the web app
python app.py
```

Then open `http://127.0.0.1:5000/predictdata` in a browser, fill in the
form, and submit to get a predicted math score.

---

## Common Errors & How They Were Fixed

Real errors hit while building this project, kept here since recognizing
them quickly is genuinely useful:

| Error | Cause | Fix |
|---|---|---|
| `TypeError: __init__() got an unexpected keyword argument 'stratergy'` | Misspelled `strategy` keyword in `SimpleImputer(...)`. | Match the exact parameter name. |
| `ValueError: Cannot center sparse matrix: pass with_mean=False` | `StandardScaler()` applied right after `OneHotEncoder`, whose output is a sparse matrix. | `StandardScaler(with_mean=False)` in that pipeline. |
| `ImportError: cannot import name 'train_test_split'` | Imported from `sklearn.preprocessing` instead of `sklearn.model_selection`. | Import from `sklearn.model_selection`. |
| `TypeError: evaluate_models() got an unexpected keyword argument 'x_test'` | Case mismatch between how a function was defined (`X_test`) and called (`x_test`). | Keyword argument names must match exactly (case-sensitive) — only matters for `name=value` calls, not positional ones. |
| `FileNotFoundError: No such file or directory: 'notebook/data/student.csv'` | Ran `data_ingestion.py` from inside `src/components/` instead of the project root — the path is relative to the current working directory. | Run from the project root: `python src/components/data_ingestion.py`, or `python -m src.components.data_ingestion`. |
| Errors silently disappearing / nothing showing up in logs | A placeholder `except: pass` block was swallowing real exceptions during early scaffolding. | Replace every placeholder with `except Exception as e: raise CustomException(e, sys)` before actually testing the code. |
| `artifacts/` still shows up in `git status` after adding it to `.gitignore` | `.gitignore` only prevents *new, untracked* files from being staged — it doesn't retroactively untrack files already committed. | `git rm -r --cached artifacts` then commit — removes it from tracking without deleting it locally. |
| Docker/Elastic Beanstalk deployment failing with both `app.py` and `application.py` present | Two similarly-structured entry-point files existed at once, conflicting with the configured `WSGIPath`. | Keep exactly one entry-point file matching the deployment config. |
| ECR image push failing in GitHub Actions | The `ECR_REPOSITORY_NAME` secret was set to the full registry URL instead of just the short repository name. | Use only the repository name in that secret; re-run the workflow from the Actions tab. |
| Prediction request timing out on EC2 | Security group only allowed HTTP/HTTPS (80/443) inbound, but the app was configured to listen on port 8080. | Add a custom TCP inbound rule for port 8080 in the EC2 security group. |

---

## Project Structure

```
mlproject/
├── .git/
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD workflow (Docker + EC2 path)
├── .gitignore
├── .ebextensions/
│   └── python.config              # Elastic Beanstalk entry-point config
├── Dockerfile
├── README.md
├── requirements.txt
├── setup.py
├── venv/                          # local environment (gitignored)
├── app.py                         # Flask entry point (local dev)
├── application.py                 # Flask entry point (deployment)
├── templates/
│   ├── index.html
│   └── home.html
├── artifacts/                     # generated, gitignored
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
├── notebook/
│   ├── data/
│   │   └── student.csv
│   ├── 1_EDA.ipynb
│   └── 2_MODEL_TRAINING.ipynb
└── src/
    ├── __init__.py
    ├── logger.py
    ├── exception.py
    ├── utils.py                   # save_object(), load_object(), evaluate_models()
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    └── pipeline/
        ├── __init__.py
        ├── train_pipeline.py
        └── predict_pipeline.py
```

---

## Why Generated Artifacts Are Not Committed

Generated files such as `artifacts/model.pkl` and `artifacts/preprocessor.pkl`
are excluded from source control. They can be recreated by running the
training pipeline after cloning the repository.

This keeps GitHub focused on the **source code, EDA notebooks, ML workflow,
hyperparameter tuning, prediction pipeline, Flask application, and deployment
configuration** that a recruiter needs to review, while keeping the project
reproducible.

```gitignore
artifacts/
```

---

## Future Improvements

- **`train_pipeline.py` is currently empty** — training is run via
  `data_ingestion.py`'s `__main__` block, which calls ingestion,
  transformation, and training in sequence. Wrapping that same sequence
  into a proper `TrainPipeline` class (mirroring how `predict_pipeline.py`
  is structured) would clean this up.
- **Move the hyperparameter grid into a config file** (e.g. YAML), loaded
  at runtime, rather than hardcoding it inside `model_trainer.py` — makes
  tuning ranges editable without touching training code.
- **Add real unit tests** in the `integration` job of the GitHub Actions
  workflow (currently a placeholder `echo` command) — e.g. tests for
  `evaluate_models()`, `get_data_transformer_object()`, and the Flask
  routes.
- **Add data validation** as its own component — checking incoming data
  against an expected schema before ingestion proceeds.
- **Read from a real data source** (MongoDB, an API, a data warehouse)
  instead of a static local CSV, changing only the one line inside
  `initiate_data_ingestion()` that currently calls `pd.read_csv(...)`.
- **Turn `debug=True` off by default** in `app.py`, keeping it opt-in via
  an environment variable rather than hardcoded — reduces the risk of it
  accidentally shipping to a public deployment.
