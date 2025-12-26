from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app import compute_total_fare_by_class, load_data


app = FastAPI(
    title="Titanic ML API",
    description="API lab7",
    version="1.0.0",
)


DATA_PATH = Path(__file__).resolve().parent / "titanic_train.csv"
MODEL_PATH = Path(__file__).resolve().parent / "titanic_model.joblib"

# Глобальная переменная для хранения модели
ml_model = None
label_encoders = {}


class TotalFareRequest(BaseModel):
    sex: str


class TotalFareResponseItem(BaseModel):
    Класс_обслуживания: int
    Суммарная_стоимость_билетов: float


class TotalFareResponse(BaseModel):
    sex: str
    items: list[TotalFareResponseItem]


class PredictSurvivalRequest(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


class PredictSurvivalResponse(BaseModel):
    survived: int
    probability: float


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовка признаков для обучения модели."""
    df = df.copy()

    # Заполнение пропущенных значений
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Кодирование категориальных признаков
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df["Sex_encoded"] = le_sex.fit_transform(df["Sex"])
    df["Embarked_encoded"] = le_embarked.fit_transform(df["Embarked"])

    # Сохранение энкодеров
    label_encoders["Sex"] = le_sex
    label_encoders["Embarked"] = le_embarked

    # Выбор признаков
    features = ["Pclass", "Sex_encoded", "Age", "SibSp", "Parch", "Fare", "Embarked_encoded"]
    return df[features]


def train_model() -> dict:
    """Обучение модели машинного обучения для предсказания выживания."""
    global ml_model

    if not DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Файл с данными не найден.")

    df = load_data(str(DATA_PATH))

    if "Survived" not in df.columns:
        raise HTTPException(status_code=500, detail="Колонка 'Survived' не найдена в данных.")

    # Подготовка данных
    X = prepare_features(df)
    y = df["Survived"]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка точности
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Сохранение модели
    ml_model = model
    joblib.dump(model, MODEL_PATH)

    return {
        "status": "success",
        "train_accuracy": float(train_score),
        "test_accuracy": float(test_score),
        "message": "Модель успешно обучена и сохранена",
    }


@app.post("/total_fare_by_class", response_model=TotalFareResponse)
def total_fare_by_class(payload: TotalFareRequest) -> TotalFareResponse:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Файл с данными не найден.")

    df: pd.DataFrame = load_data(str(DATA_PATH))

    result_df = compute_total_fare_by_class(df, payload.sex)

    items = [
        TotalFareResponseItem(
            Класс_обслуживания=int(row["Класс обслуживания"]),
            Суммарная_стоимость_билетов=float(row["Суммарная стоимость билетов"]),
        )
        for _, row in result_df.iterrows()
    ]

    return TotalFareResponse(sex=payload.sex, items=items)


@app.post("/train_model")
def train_ml_model() -> dict:
    """Эндпоинт для обучения модели машинного обучения."""
    try:
        result = train_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели: {str(e)}")


@app.post("/predict_survival", response_model=PredictSurvivalResponse)
def predict_survival(payload: PredictSurvivalRequest) -> PredictSurvivalResponse:
    """Эндпоинт для предсказания выживания пассажира."""
    global ml_model

    # Загрузка модели, если она не загружена
    if ml_model is None:
        if MODEL_PATH.exists():
            ml_model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=400,
                detail="Модель не обучена. Сначала вызовите /train_model",
            )

    # Подготовка данных для предсказания
    try:
        # Кодирование категориальных признаков
        if "Sex" not in label_encoders or "Embarked" not in label_encoders:
            # Если энкодеры не загружены, нужно обучить модель заново
            train_model()

        le_sex = label_encoders["Sex"]
        le_embarked = label_encoders["Embarked"]

        sex_encoded = le_sex.transform([payload.Sex])[0]
        embarked_encoded = le_embarked.transform([payload.Embarked])[0]

        # Формирование признаков
        features = [
            [
                payload.Pclass,
                sex_encoded,
                payload.Age,
                payload.SibSp,
                payload.Parch,
                payload.Fare,
                embarked_encoded,
            ]
        ]

        # Предсказание
        prediction = ml_model.predict(features)[0]
        probability = ml_model.predict_proba(features)[0][1]  # Вероятность выживания

        return PredictSurvivalResponse(survived=int(prediction), probability=float(probability))
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка при обработке данных: {str(e)}. Убедитесь, что значения Sex и Embarked корректны.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")
