
from django.shortcuts import render

from .forms import StudentMarksForm

from predict import predict_marks


def home(request):

    prediction = None
    error = None

    # Check whether the form was submitted
    if request.method == "POST":

        # Collect submitted values
        form = StudentMarksForm(request.POST)

        # Check input validity
        if form.is_valid():

            # Get validated inputs
            data = form.cleaned_data

            study_hours = data["study_hours"]
            attendance = data["attendance"]
            previous_marks = data["previous_marks"]

            try:
                # Call the trained ML model
                prediction = predict_marks(
                    study_hours,
                    attendance,
                    previous_marks
                )

            except FileNotFoundError:
                error = (
                    "Model not found. "
                    "Please run python train.py first."
                )

            except (ValueError, OSError):
                error = "Unable to generate prediction."

    else:
        # Display an empty form on the first visit
        form = StudentMarksForm()

    # Return the webpage
    return render(
        request,
        "predictor/home.html",
        {
            "form": form,
            "prediction": prediction,
            "error": error
        }
    )
