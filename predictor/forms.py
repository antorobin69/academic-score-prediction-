
from django import forms


class StudentMarksForm(forms.Form):

    study_hours = forms.FloatField(
        label="Study Hours Per Day",
        min_value=0,
        max_value=12,
        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "placeholder": "Enter study hours"
            }
        )
    )

    attendance = forms.FloatField(
        label="Attendance Percentage",
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "placeholder": "Enter attendance"
            }
        )
    )

    previous_marks = forms.FloatField(
        label="Previous Exam Marks",
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "placeholder": "Enter previous marks"
            }
        )
    )
