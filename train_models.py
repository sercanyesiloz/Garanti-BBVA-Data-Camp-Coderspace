import gc
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from utils import plot_importances

def get_model_scores(model, splitter, train_set, test_set, target: str, n_folds: int, plot_imp: bool = False):

    X = train_set.drop(columns=[target], axis=1).copy()
    y = train_set[target].copy()
    features = X.columns

    X_test = test_set.drop(columns=[target], axis=1).copy()
    y_oof = np.zeros(X.shape[0])
    y_oof_score = np.zeros((X.shape[0], 2))
    y_pred = np.zeros(X_test.shape[0])
    y_score = np.zeros((X_test.shape[0], 2))

    scores = list()
    for idx, (train_ind, val_ind) in enumerate(splitter.split(X, y)):
        print(f"| Fold {idx+1} |".center(80, "-"))
        X_train = X.iloc[train_ind]
        y_train = y.iloc[train_ind]
        X_val = X.iloc[val_ind]
        y_val = y.iloc[val_ind]
        print(f'train: {X_train.shape}')
        print(f'val: {X_val.shape}')

        #Training Classifier
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=500,
            verbose=250,
        )

        #Feature Importances        
        if plot_imp:
            plot_importances(model, features)

        #Validation Predictions
        val_pred = model.predict(X_val)
        y_oof[val_ind] += val_pred

        val_score = model.predict_proba(X_val)
        y_oof_score[val_ind] += val_score

        #Test Predictions
        test_pred = model.predict(X_test)
        y_pred += test_pred / n_folds

        test_score = model.predict_proba(X_test)
        y_score += test_score / n_folds

        print(f'fold accuracy: {accuracy_score(y_val, val_pred)}')
        scores.append(accuracy_score(y_val, val_pred))
        del X_train, y_train, X_val, y_val
        gc.collect()

    print(f'accuracy: {accuracy_score(y, y_oof)}')
    print(f'folds avg accuracy: {np.mean(scores)}')
    
    return y_score, y_oof_score