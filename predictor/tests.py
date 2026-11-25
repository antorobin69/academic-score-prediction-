
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from .forms import StudentMarksForm


# TEST 1: Form validation tests
class StudentFormTests(TestCase):

    def test_valid_student_details(self):

        form = StudentMarksForm(data={
            "study_hours": 5,
            "attendance": 85,
            "previous_marks": 70
        })

        self.assertTrue(form.is_valid())


    def test_invalid_attendance(self):

        form = StudentMarksForm(data={
            "study_hours": 5,
            "attendance": 150,
            "previous_marks": 70
        })

        self.assertFalse(form.is_valid())

        self.assertIn(
            "attendance",
            form.errors
        )


    def test_invalid_study_hours(self):

        form = StudentMarksForm(data={
            "study_hours": -3,
            "attendance": 85,
            "previous_marks": 70
        })

        self.assertFalse(form.is_valid())

        self.assertIn(
            "study_hours",
            form.errors
        )


# TEST 2: Django webpage tests
class StudentPageTests(TestCase):

    def test_home_page(self):

        url = reverse("predictor:home")

        response = self.client.get(url)

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertContains(
            response,
            "Student Marks Prediction"
        )


    @patch(
        "predictor.views.predict_marks",
        return_value=72.5
    )
    def test_prediction_display(self, mock_prediction):

        url = reverse("predictor:home")

        response = self.client.post(
            url,
            {
                "study_hours": 5,
                "attendance": 85,
                "previous_marks": 70
            }
        )

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertContains(
            response,
            "72.5"
        )

        mock_prediction.assert_called_once_with(
            5.0, 85.0, 70.0
        )


    @patch("predictor.views.predict_marks")
    def test_invalid_input_not_predicted(self, mock_prediction):

        url = reverse("predictor:home")

        response = self.client.post(
            url,
            {
                "study_hours": 5,
                "attendance": 150,
                "previous_marks": 70
            }
        )

        self.assertEqual(response.status_code, 200)

        self.assertFalse(
            response.context["form"].is_valid()
        )

        mock_prediction.assert_not_called()
