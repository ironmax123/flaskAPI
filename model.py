import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class IncomeModel:
    def __init__(self, file_path):
        # Excelファイルの読み込み
        self.data_men = self._clean_data(pd.read_excel(file_path, sheet_name="男性"))
        self.data_women = self._clean_data(pd.read_excel(file_path, sheet_name="女性"))
        self.data_overall = self._clean_data(
            pd.read_excel(file_path, sheet_name="全体")
        )
        self.data_2003_2022 = self._prepare_data()

        # 2019年のデータ
        self.data_2019 = {
            "個人所得の平均": {"男性": 463.0972, "女性": 213.8470, "全体": 349.4484},
            "正規雇用者の平均": {"男性": 516.6792, "女性": 323.5127, "全体": 451.5845},
            "非正規雇用者の平均": {
                "男性": 280.0044,
                "女性": 137.8940,
                "全体": 182.4265,
            },
            "自営業者の平均": {"男性": 395.7276, "女性": 200.1417, "全体": 351.7746},
        }
        self.model = self._train_model()

    def _clean_data(self, df):
        numeric_columns = [col for col in df.columns if "平均" in col]

        for col in numeric_columns:
            df[col] = df[col].replace(["…", "-"], np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _prepare_data(self):
        data_men = self.data_men.copy()
        data_women = self.data_women.copy()
        data_overall = self.data_overall.copy()

        data_men["性別"] = "男性"
        data_women["性別"] = "女性"
        data_overall["性別"] = "全体"

        # 列名を統一
        data_men.columns = [col.replace("男性の", "") for col in data_men.columns]
        data_women.columns = [col.replace("女性の", "") for col in data_women.columns]
        data_overall.columns = [
            col.replace("全体の", "") for col in data_overall.columns
        ]

        # データの結合
        data_combined = pd.concat(
            [data_men, data_women, data_overall], ignore_index=True
        )

        occupation_columns = [col for col in data_combined.columns if "平均" in col]

        data_combined = data_combined[["年度(総数)", "性別"] + occupation_columns]

        data_melted = data_combined.melt(
            id_vars=["年度(総数)", "性別"], var_name="職業", value_name="平均所得"
        )
        data_melted = data_melted.dropna()
        return data_melted

    def _train_model(self):
        X = self.data_2003_2022[["年度(総数)", "性別", "職業"]]
        y = self.data_2003_2022["平均所得"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["年度(総数)"]),
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ["性別", "職業"],
                ),
            ]
        )

        model_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regressor", Ridge())]
        )

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"モデルのMSE: {mse:.4f}")

        return model_pipeline

    def predict_income(self, year, gender, occupation):
        if year == 2019:
            try:
                return self.data_2019[occupation][gender]
            except KeyError:
                raise ValueError(
                    f"職業'{occupation}'または性別'{gender}'がデータに存在しません。"
                )

        input_data = pd.DataFrame(
            {"年度(総数)": [year], "性別": [gender], "職業": [occupation]}
        )

        try:
            predicted_income = self.model.predict(input_data)
            return predicted_income[0]
        except Exception as e:
            raise ValueError(f"予測中にエラーが発生しました: {str(e)}")


# メイン関数
def modls():
    file_path = "所得【男女別雇用別平均】.xlsx"  # Excelファイルのパスを指定
    income_model = IncomeModel(file_path)

    genders = ["男性", "女性", "全体"]
    occupations = [
        "個人所得の平均",
        "正規雇用者の平均",
        "非正規雇用者の平均",
        "自営業者の平均",
    ]

    print("\n予測結果:")
    for year in [2019, 2020, 2021, 2022, 2023, 2024]:
        for gender in genders:
            for occupation in occupations:
                try:
                    predicted_value = income_model.predict_income(
                        year, gender, occupation
                    )
                    return (
                        f"{year}年の{gender}の{occupation}の予測値: "
                        f"{predicted_value:.4f}"
                    )
                except ValueError as e:
                    print(f"予測エラー: {year}年の{gender}の{occupation} - {e}")
