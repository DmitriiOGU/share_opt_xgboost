```
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold,StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna.samplers import TPESampler

def objective(trial: optuna.Trial) -> float:
    global X, y
    param = {
        "verbosity": 0,
        "objective": trial.suggest_categorical("objective",["binary:logistic","binary:logitraw","binary:hinge"]),
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "scale_pos_weight": 50.8311688311688
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 30, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    skf = StratifiedKFold(5, shuffle=True, random_state=33)
    f1_list = []
    acc_list = []
    for train_index, test_index in skf.split(X,y):
        gbm = XGBClassifier(**param)
        gbm.fit(
            X[train_index],
            y[train_index],
            verbose=0,
        )
        preds = gbm.predict(X[test_index])
        pred_labels = np.rint(preds)
        f1_list.append(f1_score(y[test_index], pred_labels))
        acc_list.append(accuracy_score(y[test_index], pred_labels))
    return mean(f1_list), mean(acc_list)
def train_optuna_func():
    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study( directions=["maximize","maximize"],sampler=sampler)
    study.optimize(objective, n_trials=120)
    trial =  max(study.best_trials, key=lambda i: i.values[0])
    print(best)
    print("  Value: {}".format(trial.values))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
train_optuna_func()
```
