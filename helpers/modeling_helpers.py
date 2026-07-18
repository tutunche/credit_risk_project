import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from xgboost import XGBClassifier





def train_lgbm(
    df: pd.DataFrame,
    seed,
    target: str = "TARGET",
    n_splits: int = 5,
    params: dict | None = None,
):
    """Cross-validated LightGBM AUC for measuring feature lift.

    Runs seeded StratifiedKFold, trains one model per fold (early stopping on
    the fold's own validation slice), and scores AUC on that slice. Returns the
    mean and std of the per-fold AUCs. Compare two calls with the same seed to
    test the lift from adding a variable, and judge the lift against the std.
    """
    y = df[target]
    X = df.drop(columns=[target, "SK_ID_CURR"])
    default_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 3000,
        "verbosity": -1,
        "random_state": seed,
    }
    if params:
        default_params.update(params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    i=0
    for tr_idx, val_idx in skf.split(X, y):
        print(i)
        i += 1
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        

        num_cols = X_tr.select_dtypes(include=np.number).columns.to_list()
        cat_cols = X_tr.columns.difference(num_cols).to_list()

        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ])

        ct = ColumnTransformer([
            ("cat", cat_pipe, cat_cols),
        ],
        remainder='passthrough'
        )

        X_tr_processed = ct.fit_transform(X_tr)
        X_val_processed = ct.transform(X_val)
        model = lgb.LGBMClassifier(**default_params)
        model.fit(
            X_tr_processed, y_tr,
            eval_X=X_val_processed,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_pred = model.predict_proba(X_val_processed)[:, 1]
        fold_aucs.append(roc_auc_score(y_val, val_pred))

    fold_aucs = np.array(fold_aucs)
    return {
        "auc_mean": fold_aucs.mean(),
        "auc_std": fold_aucs.std(),
        "fold_aucs": fold_aucs,
    }

def get_xgb_pipeline(cat_cols:list):
    preprocess_xgb = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder='passthrough'
    )
    xgb_pipe = Pipeline([
        ("preprocess", preprocess_xgb),
        ("xgb_model", XGBClassifier())
    ])
    return xgb_pipe

class NNModel(nn.Module):
    def __init__(self, in_dims, dropout_rate=0.2):
        super().__init__()
        
        self.fc1 = nn.Linear(in_dims, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x =self.output(x)
        return x
    


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, in_dims: int, dropout_rate: float=0.2, lr: float=1e-3, weight_decay: float=1e-5,
                 batch_size=64, max_epochs=10):
        self.in_dims = in_dims
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.unique(y)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")




        X_tensor = torch.tensor(X, dtype=torch.float32).to(dev)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(dev).reshape(-1, 1)

        n_pos = y_tensor.sum()
        n_neg = y_tensor.numel() - n_pos
        pos_weight = n_neg / n_pos

        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = NNModel(X.shape[1], self.dropout_rate).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



        for epoch in tqdm(range(self.max_epochs)):
            self.model_.train()
            for xb, yb in tqdm(data_loader, leave=False):
                y_hat = self.model_(xb)
                loss = loss_fn(y_hat, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_set_Tensor = torch.tensor(X, dtype=torch.float32).to(dev)
        X_set = TensorDataset(X_set_Tensor)
        X_loader = DataLoader(X_set, batch_size=64, shuffle=False)

        all_proba = []
        with torch.no_grad():
            self.model_.eval()
            for xb in X_loader:
                xb = xb[0]
                y_hat = torch.sigmoid(self.model_(xb))
                all_proba.append(y_hat.cpu().numpy())
        
        p = np.concatenate(all_proba).reshape(-1)
        return np.column_stack([1 - p, p])


    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

