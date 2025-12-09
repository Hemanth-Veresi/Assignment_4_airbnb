from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def evaluate_model(model, X_test, y_test, is_keras=False):
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : fitted model (XGBoost or Keras)
    X_test : np.array or DataFrame
    y_test : Series or array
    is_keras : bool
        If True, use model.predict() from Keras, otherwise assume
        the model behaves like scikit-learn (XGBoost).

    Returns
    -------
    dict : { 'rmse', 'mae', 'r2' }
    """

    # Get predictions
    if is_keras:
        preds = model.predict(X_test).reshape(-1)
    else:
        preds = model.predict(X_test)

    # Compute evaluation metrics
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def compare_models(results):
    """
    Convert a dictionary of model results into a DataFrame for easy viewing.

    Example input:
    {
        "xgb": {"rmse": ..., "mae": ..., "r2": ...},
        "nn_small": {...},
        "nn_deep": {...}
    }

    Returns a pandas DataFrame.
    """

    df = pd.DataFrame(results).T
    df.index.name = "model"
    return df
