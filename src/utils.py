import pandas as pd

def load_data(filepath):
    """Load CSV data from file."""
    return pd.read_csv(filepath)

def save_to_csv(df, filepath):
    """Save DataFrame to CSV."""
    df.to_csv(filepath, index=False)
