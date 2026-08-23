# ============================================================
# FLASK APPLICATION
# ============================================================
#
# PURPOSE:
# This file connects our HTML frontend with the machine
# learning prediction pipeline.
#
# COMPLETE FLOW:
#
# User opens website
#        ↓
#     Flask
#        ↓
#     HTML Form
#        ↓
#    User submits data
#        ↓
#     app.py
#        ↓
#    CustomData
#        ↓
#     DataFrame
#        ↓
# PredictPipeline
#        ↓
# preprocessor.pkl
#        ↓
#    model.pkl
#        ↓
#    Prediction
#        ↓
#     home.html
#
# ============================================================


from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================
#
# Flask(__name__) creates our Flask application.
#
# ============================================================

application = Flask(__name__)


# ============================================================
# APP VARIABLE
# ============================================================
#
# Create an "app" variable that points to the Flask
# application.
#
# This is commonly used when running/deploying Flask apps.
#
# ============================================================

app = application


# ============================================================
# HOME PAGE ROUTE
# ============================================================
#
# URL:
#
#     /
#
# Example:
#
#     http://127.0.0.1:5000/
#
# When the user visits this URL, Flask renders index.html.
#
# ============================================================


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================
#
# URL:
#
#     /predictdata
#
# This route accepts:
#
#     GET
#     POST
#
# GET:
#     Display the prediction form.
#
# POST:
#     Receive form data and make a prediction.
#
# ============================================================


@app.route(
    "/predictdata",
    methods=["GET", "POST"]
)
def predict_datapoint():

    # ========================================================
    # GET REQUEST
    # ========================================================
    #
    # When the user opens the prediction page, the browser
    # sends a GET request.
    #
    # We display home.html.
    #
    # ========================================================

    if request.method == "GET":

        return render_template("home.html")

    # ========================================================
    # POST REQUEST
    # ========================================================
    #
    # When the user submits the form, the browser sends
    # a POST request.
    #
    # The submitted values can be accessed using:
    #
    #     request.form.get()
    #
    # ========================================================

    else:

        # ====================================================
        # STEP 1: GET DATA FROM HTML FORM
        # ====================================================
        #
        # The "name" attribute from the HTML form must match
        # the names used here.
        #
        # Example:
        #
        # HTML:
        #
        # <input name="reading_score">
        #
        # Python:
        #
        # request.form.get("reading_score")
        #
        # ====================================================

        data = CustomData(

            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            # IMPORTANT:
            #
            # Reading score must come from
            # the reading_score field.
            #
            reading_score=float(request.form.get("reading_score")),

            # Writing score must come from
            # the writing_score field.
            #
            writing_score=float(request.form.get("writing_score"))
        )

        # ====================================================
        # STEP 2: CONVERT USER INPUT INTO DATAFRAME
        # ====================================================
        #
        # CustomData stores the user input.
        #
        # get_data_as_data_frame() converts the input into
        # a Pandas DataFrame.
        #
        # ====================================================

        pred_df = data.get_data_as_data_frame()

        # Print the DataFrame in the terminal.
        #
        # This is useful while learning and debugging.

        print("\nInput Data:")
        print(pred_df)

        print("\nBefore Prediction")

        # ====================================================
        # STEP 3: CREATE PREDICTION PIPELINE
        # ====================================================
        #
        # PredictPipeline is responsible for:
        #
        #     1. Loading preprocessor.pkl
        #     2. Transforming the input
        #     3. Loading model.pkl
        #     4. Making the prediction
        #
        # ====================================================

        predict_pipeline = PredictPipeline()
        print("Prediction Pipeline Created")

        # ====================================================
        # STEP 4: MAKE PREDICTION
        # ====================================================
        #
        # Pass the DataFrame to PredictPipeline.
        #
        # Internally:
        #
        #     DataFrame
        #          ↓
        #     Preprocessor
        #          ↓
        #     Transformed Data
        #          ↓
        #     Trained Model
        #          ↓
        #     Prediction
        #
        # ====================================================

        results = predict_pipeline.predict(pred_df)

        print("After Prediction")

        # ====================================================
        # STEP 5: DISPLAY RESULT
        # ====================================================
        #
        # model.predict() normally returns a NumPy array.
        #
        # Example:
        #
        #     results = [78.45]
        #
        # results[0] gives:
        #
        #     78.45
        #
        # We send this value to home.html.
        #
        # ====================================================

        return render_template("home.html",results=results[0])

# ============================================================
# RUN FLASK APPLICATION
# ============================================================
#
# This block runs only when we execute:
#
#     python app.py
#
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0"
    )