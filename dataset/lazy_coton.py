# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LarsCV, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Chargement
df = pd.read_csv("dataset_complet_2000_2026.csv")
target = "Rdt_region_kg_ha"
X = df.drop(columns=[target, 'Année'])
y = df[target]

# Encodage one-hot
cat_cols = ['Région', 'Texture', 'Variete']
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# GroupKFold par année
groups = df['Année']
gkf = GroupKFold(n_splits=5)

# Modèles
models = {
    "LarsCV": Pipeline([('scaler', StandardScaler()), ('model', LarsCV(cv=5))]),
    "RidgeCV": Pipeline([('scaler', StandardScaler()), ('model', RidgeCV(alphas=[0.1,1,10,50,100,500], cv=5))]),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
}

# Évaluation
scores = {}
for name, model in models.items():
    r2_list = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_list.append(r2_score(y_test, y_pred))
    scores[name] = np.mean(r2_list)
    print(f"{name}: R² moyen = {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# Résultat attendu : RandomForest ~0.79, LarsCV ~0.60