Hai!
Lebih jelas lihat [ini](https://gist.github.com/Qfl3x/ccff6b0708358c040e437d52af0c2e43)

# Virtual Environment
Untuk membuat python virtual environment, bisa gunakan pyenv-virtualenv atau conda.
```
pyenv virtualenv 3.8.5 mlOps
```

Jika ingin install requirements.txt, bisa gunakan perintah berikut:
```
pyenv activate mlOps
pip install -r requirements.txt
```

# MLFlow UI with backend
```
mlflow server --backend-store-uri sqlite:///mlflow.db 
```
Apa kegunaanya? Ketika tidak menggunakan backend, maka setiap kali mlflow server dijalankan, maka akan membuat database baru. Jika menggunakan backend, maka mlflow server akan mengakses database yang sudah ada.

# Menambahkan MLFlow Tracking pada Project
1. import mlflow
2. Atur tracking uri dan set experiment
```
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")
```
3. Pada bagian yang ingin di tracking, tambahkan `with mlflow.start_run(): {code}`. Pada bagian ini juga, bisa tambahkan `mlflow.log_param("param_name", param_value)` dan `mlflow.log_metric("metric_name", metric_value)`, atau bahkan artifact dengan `mlflow.log_artifact("artifact_path")` 

# Hyperparameter Tuning with hyperopt
1. Import library yang dibutuhkan
```
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
```
2. Pastikan data sudah di load dan di split menjadi train dan test
```
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)
```
3. Define objective function
```
def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}
```
4. Define hyperparameter space
```
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}
```
5. Passing objective function and hyperparameter space to fmin
```
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
```
6. Cek hasilnya.

# Autologging
1. Tambahkan `mlflow.xgboost.autolog()` sebelum training di `with mlflow.start_run():`

# Model Management (add artifact)
1. tambahkan `mlflow.log_artifact(local_path, artifact_path)` pada bagian yang ingin dijadikan artifact. Atau,
2. `mlflow.{type}.log_model(model, artifact_path)` untuk menyimpan model. Contoh: `mlflow.xgboost.log_model(booster, "model")`
3. Dari artifact itu, nanti bisa di load kembali dengan `mlflow.{type}.load_model(artifact_path)`. Contoh: `mlflow.xgboost.load_model("model")` (ada panduannya di artifact section mlflow)

