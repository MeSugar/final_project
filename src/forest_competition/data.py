from pathlib import Path
from typing import Tuple

import click
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def generate_features(df : pd.DataFrame) -> pd.DataFrame:
    df['EVDtH'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['EHDtH'] = df['Elevation'] - df['Horizontal_Distance_To_Hydrology']*0.15
    df['EDtH'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
    return df

def get_data(
    dataset_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(dataset_path, index_col='Id')
    click.echo(f"Dataset shape: {data.shape}.")
    X = data.drop("Cover_Type", axis=1)
    y = data["Cover_Type"]
    return X, y