"""Data processing utilities for handling dataset operations."""

import pandas as pd
from config import SAMPLE_RANDOM_STATE


class DataProcessor:
    def __init__(self):
        pass

    def load_csv(self, uploaded_file):
        """Load CSV file and return dataframe."""
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}"

    def sample_large_dataset(self, df: pd.DataFrame, max_rows: int):
        """Sample large datasets for analysis."""
        if len(df) > max_rows:
            return df.sample(max_rows, random_state=SAMPLE_RANDOM_STATE), True
        return df, False

    def get_dataframe_info(self, df: pd.DataFrame):
        """Extract comprehensive information about the dataframe."""
        # Convert dtypes to strings to avoid JSON serialization issues
        dtypes_str = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        return {
            "columns": df.columns.tolist(),
            "dtypes": dtypes_str,
            "shape": df.shape,
            "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
            "categorical_columns": df.select_dtypes(include='object').columns.tolist()
        }

    def get_missing_values_info(self, df: pd.DataFrame):
        """Get information about missing values in the dataset."""
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        return missing_df[missing_df['Missing'] > 0]

    def get_column_types_info(self, df: pd.DataFrame):
        """Get information about column types."""
        return pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values.astype(str)
        })