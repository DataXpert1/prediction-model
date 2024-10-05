import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def create_lag_features(df, columns, lags=3):
    """Create lagged features for time-series data."""
    for col in columns:
        for lag in range(1, lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    df.fillna(method='bfill', inplace=True)
    return df

def engineer_features(df):
    """Create new features such as growth rates and ratios."""
    df['revenue_growth'] = df['revenue_2024'].pct_change()
    df['expense_growth'] = df['expense_2024'].pct_change()
    return df

def preprocess_data(df):
    """Handle missing values, encoding, and scaling."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False)
    df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))

    df.drop(categorical_columns, axis=1, inplace=True)
    df = pd.concat([df, df_encoded], axis=1)

    # Scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def split_data(df, target_columns):
    """Split data into training and testing sets."""
    X = df.drop(target_columns, axis=1)
    y = df[target_columns]
    return train_test_split(X, y, test_size=0.2, random_state=42)
