from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv("heart_disease_500_patients.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = pd.DataFrame([{
        "age": int(request.form["age"]),
        "sex": int(request.form["sex"]),
        "cp": int(request.form["cp"]),
        "trestbps": int(request.form["trestbps"]),
        "chol": int(request.form["chol"]),
        "fbs": int(request.form["fbs"]),
        "restecg": int(request.form["restecg"]),
        "thalach": int(request.form["thalach"]),
        "exang": int(request.form["exang"]),
        "oldpeak": float(request.form["oldpeak"]),
        "slope": int(request.form["slope"]),
        "ca": int(request.form["ca"]),
        "thal": int(request.form["thal"])
    }])

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    result = "Patient is likely to have Heart Disease ❤️" \
        if prediction[0] == 1 else \
        "Patient is unlikely to have Heart Disease 💚"

    return render_template(
        "result.html",
        result=result,
        no=round(probability[0][0]*100,2),
        yes=round(probability[0][1]*100,2)
    )


if __name__ == "__main__":
    app.run(debug=True)