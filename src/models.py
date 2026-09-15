import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold


class AdaptivePhishFusionMLP(nn.Module):
    """
    Sample-Adaptive Multi-Branch Neural Network for Multimodal Phishing Detection.
    Computes sample-specific Softmax attention weights [g_url(x), g_domain(x), g_html(x)]
    dynamically for each website input x rather than using static fusion weights.
    """
    def __init__(self, url_dim=12, domain_dim=9, html_dim=9, hidden_dim=32, emb_dim=16, dropout=0.2):
        super(AdaptivePhishFusionMLP, self).__init__()

        # Sub-network Branch 1: URL / Address features
        self.url_branch = nn.Sequential(
            nn.Linear(url_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        # Sub-network Branch 2: Domain & Security features
        self.domain_branch = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        # Sub-network Branch 3: Page & HTML Behavior features
        self.html_branch = nn.Sequential(
            nn.Linear(html_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )

        # Dynamic Sample-Adaptive Attention Gating Network
        total_emb_dim = emb_dim * 3  # 16 + 16 + 16 = 48
        self.attention_gate = nn.Sequential(
            nn.Linear(total_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)  # Outputs [g_url, g_domain, g_html] summing to 1.0
        )

        # Classifier Head receiving adaptively weighted representations
        self.classifier_head = nn.Sequential(
            nn.Linear(total_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_url, x_domain, x_html, return_attention=False):
        h_url = self.url_branch(x_url)
        h_domain = self.domain_branch(x_domain)
        h_html = self.html_branch(x_html)

        # Concatenate unweighted embeddings for gating network
        h_concat = torch.cat((h_url, h_domain, h_html), dim=1)
        attn_weights = self.attention_gate(h_concat)  # (N, 3)

        g_url = attn_weights[:, 0:1]
        g_domain = attn_weights[:, 1:2]
        g_html = attn_weights[:, 2:3]

        # Apply dynamic sample-adaptive weighting to representations
        fused = torch.cat((g_url * h_url, g_domain * h_domain, g_html * h_html), dim=1)
        out = self.classifier_head(fused)
        prob = torch.sigmoid(out)

        if return_attention:
            return prob, attn_weights
        return prob


class EarlyFusionNeuralMLP(nn.Module):
    """Standard Single-Branch Early Fusion Neural MLP."""
    def __init__(self, input_dim=30, hidden_dim=64, dropout=0.2):
        super(EarlyFusionNeuralMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class PyTorchNeuralClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-Learn compatible wrapper for PyTorch Adaptive & Fixed Multimodal MLPs."""
    def __init__(self, is_multimodal=True, is_adaptive=True, epochs=35, batch_size=64, lr=0.001, weight_decay=1e-4):
        self.is_multimodal = is_multimodal
        self.is_adaptive = is_adaptive
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        if self.is_multimodal:
            x_url = X_arr[:, :12]
            x_domain = X_arr[:, 12:21]
            x_html = X_arr[:, 21:]
            dataset = TensorDataset(
                torch.tensor(x_url),
                torch.tensor(x_domain),
                torch.tensor(x_html),
                torch.tensor(y_arr)
            )
            self.model = AdaptivePhishFusionMLP()
        else:
            dataset = TensorDataset(torch.tensor(X_arr), torch.tensor(y_arr))
            self.model = EarlyFusionNeuralMLP()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                if self.is_multimodal:
                    bu, bd, bh, by = batch
                    preds = self.model(bu, bd, bh)
                else:
                    bx, by = batch
                    preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            if self.is_multimodal:
                x_url = torch.tensor(X_arr[:, :12])
                x_domain = torch.tensor(X_arr[:, 12:21])
                x_html = torch.tensor(X_arr[:, 21:])
                probs = self.model(x_url, x_domain, x_html).numpy()
            else:
                probs = self.model(torch.tensor(X_arr)).numpy()
        probs = probs.reshape(-1, 1)
        return np.hstack([1.0 - probs, probs])

    def predict_attention_weights(self, X):
        """Returns sample-level attention weights [g_url, g_domain, g_html] if adaptive."""
        if not (self.is_multimodal and self.is_adaptive):
            return np.ones((len(X), 3)) / 3.0

        self.model.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            x_url = torch.tensor(X_arr[:, :12])
            x_domain = torch.tensor(X_arr[:, 12:21])
            x_html = torch.tensor(X_arr[:, 21:])
            _, weights = self.model(x_url, x_domain, x_html, return_attention=True)
        return weights.numpy()

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class LateFusionStacker(BaseEstimator, ClassifierMixin):
    """
    Late Fusion Stacking Ensemble.
    Trains separate base classifiers on each of the 3 feature modalities,
    collects out-of-fold probability predictions, and trains a Meta-Classifier.
    """
    def __init__(self, base_estimator=None, meta_estimator=None, n_splits=5):
        self.base_estimator = base_estimator or XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
        self.meta_estimator = meta_estimator or LogisticRegression(C=1.0, random_state=42)
        self.n_splits = n_splits
        self.modality_indices = {
            "url": list(range(0, 12)),
            "domain": list(range(12, 21)),
            "html": list(range(21, 30))
        }

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        self.models_ = {}
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X_arr), 3))

        for idx, (mod_name, mod_cols) in enumerate(self.modality_indices.items()):
            X_mod = X_arr[:, mod_cols]
            oof_preds = np.zeros(len(X_arr))

            for train_idx, val_idx in skf.split(X_mod, y_arr):
                fold_model = clone(self.base_estimator)
                fold_model.fit(X_mod[train_idx], y_arr[train_idx])
                oof_preds[val_idx] = fold_model.predict_proba(X_mod[val_idx])[:, 1]

            meta_features[:, idx] = oof_preds
            full_model = clone(self.base_estimator)
            full_model.fit(X_mod, y_arr)
            self.models_[mod_name] = full_model

        self.meta_model_ = clone(self.meta_estimator)
        self.meta_model_.fit(meta_features, y_arr)
        self.classes_ = np.unique(y)
        return self

    def _get_meta_features(self, X):
        X_arr = np.asarray(X)
        meta_features = np.zeros((len(X_arr), 3))
        for idx, (mod_name, mod_cols) in enumerate(self.modality_indices.items()):
            X_mod = X_arr[:, mod_cols]
            meta_features[:, idx] = self.models_[mod_name].predict_proba(X_mod)[:, 1]
        return meta_features

    def predict_proba(self, X):
        meta_feats = self._get_meta_features(X)
        return self.meta_model_.predict_proba(meta_feats)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def get_all_models():
    """Returns a dictionary of all individual models and fusion architectures."""
    return {
        # Proposed Adaptive Multimodal Fusion Network
        "Sample-Adaptive Neural Fusion (Proposed)": PyTorchNeuralClassifier(is_multimodal=True, is_adaptive=True, epochs=35, batch_size=64, lr=0.001),
        
        # Base Classifiers
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_state=42, verbose=0),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        
        # Multimodal Fusion Baselines
        "Early Fusion (Dense Neural MLP)": PyTorchNeuralClassifier(is_multimodal=False, is_adaptive=False, epochs=35, batch_size=64, lr=0.001),
        "Late Fusion (Stacking Ensemble)": LateFusionStacker()
    }
