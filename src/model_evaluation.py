from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

def load_model(filepath):
    """Load a trained model from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print MSE for each output."""
    predictions = model.predict(X_test)

    mse_stock_price = mean_squared_error(y_test['stock_price_2024'], predictions[:, 0])
    mse_market_share = mean_squared_error(y_test['market_share_2024'], predictions[:, 1])
    mse_revenue = mean_squared_error(y_test['revenue_2024'], predictions[:, 2])
    mse_expense = mean_squared_error(y_test['expense_2024'], predictions[:, 3])

    print(f'MSE for Stock Price: {mse_stock_price}')
    print(f'MSE for Market Share: {mse_market_share}')
    print(f'MSE for Revenue: {mse_revenue}')
    print(f'MSE for Expense: {mse_expense}')

    return np.mean([mse_stock_price, mse_market_share, mse_revenue, mse_expense])
