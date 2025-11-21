import streamlit as st
import pandas as pd
from pathlib import Path


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    st.title("Пассажиры Титаника: суммарная стоимость билетов по классам")

    data_path = Path("titanic_train.csv")

    df = load_data(str(data_path))

    st.subheader("Фильтрация по полу пассажиров")

    sexes = df["Sex"].dropna().unique()
    sexes = sorted(sexes.tolist())

    selected_sex = st.selectbox("Выберите пол пассажиров", options=sexes, index=0)

    filtered_df = df[df["Sex"] == selected_sex]

    if filtered_df.empty:
        st.warning("Для выбранного пола нет данных.")
        st.stop()

    result = (
        filtered_df.groupby("Pclass", as_index=False)["Fare"]
        .sum()
        .rename(columns={"Pclass": "Класс обслуживания", "Fare": "Суммарная стоимость билетов"})
    )

    st.subheader("Суммарная стоимость билетов по классам обслуживания")
    st.dataframe(result, use_container_width=True)


if __name__ == "__main__":
    main()


