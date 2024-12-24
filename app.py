from flask import Flask, request, jsonify
from flask_cors import CORS
from model import IncomeModel

app = Flask(__name__)
CORS(app)

file_path = "所得【男女別雇用別平均】.xlsx"
income_model = IncomeModel(file_path)


@app.route("/predict", methods=["GET"])
def predict():
    year = request.args.get("year", type=int)
    gender = request.args.get("gender", type=str)
    occupation = request.args.get("occupation", type=str)

    try:
        predicted_value = income_model.predict_income(year, gender, occupation)
        return jsonify({"prediction": predicted_value})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
