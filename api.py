from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .app import compute_total_fare_by_class, load_data


app = FastAPI(
    title="Titanic ML API",
    description="API lab7",
    version="1.0.0",
)


DATA_PATH = Path(__file__).resolve().parent / "titanic_train.csv"


class TotalFareRequest(BaseModel):
    sex: str


class TotalFareResponseItem(BaseModel):
    Класс_обслуживания: int
    Суммарная_стоимость_билетов: float


class TotalFareResponse(BaseModel):
    sex: str
    items: list[TotalFareResponseItem]


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


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


