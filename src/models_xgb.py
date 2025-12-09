import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_xgb(X_train, y_train, X_val=None, y_val=None, params=None):
    params = params or dict(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)] if X_val is not None else None,
              early_stopping_rounds=20, verbose=False)
    return model

def evaluate(model, X, y):
    pred = model.predict(X)
    rmse = mean_squared_error(y, pred, squared=False)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    return dict(rmse=rmse, mae=mae, r2=r2)
