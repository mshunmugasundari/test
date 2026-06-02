import joblib
import pandas as pd

MODEL_PATH = "models/linear_regression_pima.pkl"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35
}])

prediction = model.predict(sample)[0]
class_prediction = 1 if prediction >= 0.5 else 0

print("Raw regression output:", prediction)
print("Predicted class:", class_prediction)

if class_prediction == 1:
    print("Result: Diabetic")
else:
    print("Result: Non-diabetic")
