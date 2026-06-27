import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
import os
import base64
import hashlib
import io
import datetime
import joblib

st.set_page_config(page_title="SODECOTON - Prédiction Coton", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

# ==================== DÉFINITION DES FICHIERS ====================
DATA_FILE = "dataset_complet_2000_2026.csv"  # ou dataset_complet_10_regions_2000_2026.csv
MODEL_FILE = "saved_model.pkl"

# ==================== DIAGNOSTIC (dans la sidebar) ====================
# Cette zone s'exécute à chaque chargement et affiche les régions détectées
if os.path.exists(DATA_FILE):
    try:
        df_test = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        regions_test = sorted(df_test["Région"].unique())
        # On stocke dans st.session_state pour réutilisation
        st.session_state.regions_detectees = regions_test
        st.session_state.nb_regions_detectees = len(regions_test)
    except Exception as e:
        st.session_state.regions_detectees = []
        st.session_state.nb_regions_detectees = 0
else:
    st.session_state.regions_detectees = []
    st.session_state.nb_regions_detectees = 0

# ==================== FONCTIONS ====================
def save_model(model, biais, ref_df, colonnes_modele, data_df, X_train, y_train, perf, df_original):
    state = {
        'modele': model,
        'biais': biais,
        'ref_df': ref_df,
        'colonnes_modele': colonnes_modele,
        'data_df': data_df,
        'X_train': X_train,
        'y_train': y_train,
        'perf': perf,
        'df_original': df_original
    }
    joblib.dump(state, MODEL_FILE)

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

def format_fr(x, decimals=0, with_space=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if decimals > 0:
        entier = int(abs(x))
        dec = int(round((abs(x) - entier) * 10**decimals))
        entier_str = f"{entier:,}".replace(",", " ") if with_space else f"{entier:,}".replace(",", "")
        signe = "-" if x < 0 else ""
        return f"{signe}{entier_str},{dec:0{decimals}d}"
    else:
        val = int(round(x))
        return f"{val:,}".replace(",", " ") if with_space else f"{val:,}".replace(",", "")

def r2_badge(r2):
    if r2 >= 0.80: return "🟢", "Excellent"
    elif r2 >= 0.60: return "🟡", "Acceptable"
    elif r2 >= 0.40: return "🟠", "Modeste"
    else: return "🔴", "Faible"

# ======================== SUITE : GESTION DES UTILISATEURS ========================

USER_DB_FILE = "users.csv"
MIN_PASSWORD_LENGTH = 6

def load_users():
    """Charge la base d'utilisateurs depuis users.csv. Crée le fichier par défaut s'il n'existe pas."""
    if os.path.exists(USER_DB_FILE):
        try:
            df = pd.read_csv(USER_DB_FILE)
            # Ajout des colonnes manquantes
            if 'role' not in df.columns:
                df['role'] = 'viewer'
            df['role'] = df['role'].fillna('viewer')
            df.loc[df['username'] == 'admin', 'role'] = 'admin'
            if 'plain_password' not in df.columns:
                df['plain_password'] = ''  # valeur par défaut
                # On remplit pour admin si on connaît son hash ? pas facile, on laisse vide
            try:
                df.to_csv(USER_DB_FILE, index=False)
            except PermissionError:
                st.warning("Fichier users.csv verrouillé")
            return df
        except Exception:
            pass

    # Création du fichier par défaut avec plain_password
    default_users = pd.DataFrame({
        "username": ["admin", "guest"],
        "password_hash": [
            hashlib.sha256("sodecoton2024".encode()).hexdigest(),
            hashlib.sha256("guest123".encode()).hexdigest()
        ],
        "plain_password": ["sodecoton2024", "guest123"],
        "role": ["admin", "viewer"],
        "created_at": [str(datetime.date.today()), str(datetime.date.today())]
    })
    try:
        default_users.to_csv(USER_DB_FILE, index=False)
    except PermissionError:
        st.error("Impossible de créer users.csv.")
    return default_users

def add_user(username, password, role, users_db):
    """Ajoute un nouvel utilisateur (réservé à l'admin)."""
    if len(username.strip()) < 3:
        return False, "Nom trop court (min 3 caractères)"
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Mot de passe trop court (min {MIN_PASSWORD_LENGTH} caractères)"
    if username in users_db["username"].values:
        return False, "Ce nom d'utilisateur existe déjà"
    new_hash = hashlib.sha256(password.encode()).hexdigest()
    new_row = pd.DataFrame({
        "username": [username],
        "password_hash": [new_hash],
        "plain_password": [password],   # stockage en clair pour affichage
        "role": [role],
        "created_at": [str(datetime.date.today())]
    })
    updated_db = pd.concat([users_db, new_row], ignore_index=True)
    updated_db.to_csv(USER_DB_FILE, index=False)
    return True, "Compte créé avec succès"

def delete_user(username, users_db):
    """Supprime un utilisateur (sauf l'admin)."""
    if username == "admin":
        return False, "Impossible de supprimer le compte administrateur"
    updated_db = users_db[users_db["username"] != username]
    updated_db.to_csv(USER_DB_FILE, index=False)
    return True, f"L'utilisateur '{username}' a été supprimé"

def update_password(username, new_password, users_db):
    """Met à jour le mot de passe d'un utilisateur (pour l'admin)."""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return False, f"Mot de passe trop court (min {MIN_PASSWORD_LENGTH})"
    if username not in users_db["username"].values:
        return False, "Utilisateur inexistant"
    new_hash = hashlib.sha256(new_password.encode()).hexdigest()
    users_db.loc[users_db["username"] == username, "password_hash"] = new_hash
    users_db.loc[users_db["username"] == username, "plain_password"] = new_password
    users_db.to_csv(USER_DB_FILE, index=False)
    return True, "Mot de passe mis à jour"

def reset_password(username, new_password, users_db):
    """Réinitialise le mot de passe sans vérifier l'ancien (pour oubli)."""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return False, f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères."
    if username not in users_db["username"].values:
        return False, "Nom d'utilisateur inconnu."
    new_hash = hashlib.sha256(new_password.encode()).hexdigest()
    users_db.loc[users_db["username"] == username, "password_hash"] = new_hash
    users_db.loc[users_db["username"] == username, "plain_password"] = new_password
    users_db.to_csv(USER_DB_FILE, index=False)
    return True, "Mot de passe réinitialisé avec succès."

def check_password(username, password, users_db):
    """Vérifie le couple identifiant / mot de passe."""
    hash_input = hashlib.sha256(password.encode()).hexdigest()
    user_row = users_db[users_db["username"] == username]
    if not user_row.empty:
        return user_row.iloc[0]["password_hash"] == hash_input
    return False

def get_role(username, users_db):
    """Récupère le rôle de l'utilisateur."""
    user_row = users_db[users_db["username"] == username]
    if not user_row.empty and "role" in user_row.columns:
        role_val = user_row.iloc[0].get("role", "viewer")
        if pd.isna(role_val):
            return "viewer"
        return role_val
    return "viewer"

def get_admin_password():
    """Retourne le mot de passe actuel de l'administrateur (en clair)."""
    users_db = load_users()
    admin_row = users_db[users_db["username"] == "admin"]
    if not admin_row.empty and "plain_password" in admin_row.columns:
        pwd = admin_row.iloc[0].get("plain_password", None)
        if pwd and pd.notna(pwd):
            return str(pwd)
    return "sodecoton2024"  # fallback

# ======================== INITIALISATION DE LA SESSION ========================

users_db = load_users()
query_params = st.query_params

for key, default in [("authenticated", False), ("username", ""), ("user_role", "viewer")]:
    if key not in st.session_state:
        st.session_state[key] = default

# Variables pour le message de réinitialisation
if "reset_done" not in st.session_state:
    st.session_state.reset_done = False
if "reset_message" not in st.session_state:
    st.session_state.reset_message = ""

if query_params.get("auth") == "true":
    st.session_state.authenticated = True
    st.query_params.clear()
    st.rerun()

# ======================== PAGE DE CONNEXION ========================

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("LOGO SODEC.png"):
            st.image("LOGO SODEC.png", width=80)
        st.markdown("<h1 style='text-align:center; color:#1b5e20;'>SODECOTON</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#4a4a4a;'>Module de Prédiction du Rendement Cotonnier</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ---- AFFICHAGE DU MESSAGE DE RÉINITIALISATION (si présent) ----
        if st.session_state.reset_done:
            st.success(f"✅ {st.session_state.reset_message} Vous pouvez maintenant vous connecter avec votre nouveau mot de passe.")
            st.balloons()
            st.session_state.reset_done = False
            st.session_state.reset_message = ""

        # ---- Connexion ----
        st.markdown("### 🔑 Connexion")
        username_in = st.text_input("Nom d'utilisateur", key="login_user")
        password_in = st.text_input("Mot de passe", type="password", key="login_pass")
        if st.button("Se connecter", use_container_width=True):
            if username_in.strip() == "" or password_in == "":
                st.error("Veuillez remplir tous les champs.")
            elif check_password(username_in, password_in, users_db):
                st.session_state.authenticated = True
                st.session_state.username = username_in
                st.session_state.user_role = get_role(username_in, users_db)
                st.query_params["auth"] = "true"
                st.rerun()
            else:
                st.error("Identifiants incorrects.")

        st.markdown("---")

        # ---- Réinitialisation ----
        with st.expander("🔓 Mot de passe oublié ?", expanded=False):
            st.markdown("Saisissez votre nom d'utilisateur et définissez un nouveau mot de passe.")
            reset_username = st.text_input("Votre nom d'utilisateur", key="reset_user")
            reset_new_pwd = st.text_input("Nouveau mot de passe (min 6)", type="password", key="reset_new")
            reset_confirm_pwd = st.text_input("Confirmer le nouveau mot de passe", type="password", key="reset_confirm")

            if st.button("Réinitialiser", use_container_width=True):
                if not reset_username or not reset_new_pwd or not reset_confirm_pwd:
                    st.error("Tous les champs sont obligatoires.")
                elif reset_new_pwd != reset_confirm_pwd:
                    st.error("Les mots de passe ne correspondent pas.")
                elif len(reset_new_pwd) < 6:
                    st.error("Le mot de passe doit faire au moins 6 caractères.")
                else:
                    current_users = load_users()
                    if reset_username not in current_users["username"].values:
                        st.error("Nom d'utilisateur inconnu.")
                    else:
                        success, msg = reset_password(reset_username, reset_new_pwd, current_users)
                        if success:
                            st.session_state.reset_done = True
                            st.session_state.reset_message = msg
                            st.rerun()
                        else:
                            st.error(msg)
    st.stop()
def get_admin_password():
    """Retourne le mot de passe actuel de l'administrateur (en clair)."""
    USER_DB_FILE = "users.csv"
    if os.path.exists(USER_DB_FILE):
        try:
            df = pd.read_csv(USER_DB_FILE)
            if "plain_password" in df.columns:
                admin_row = df[df["username"] == "admin"]
                if not admin_row.empty:
                    pwd = admin_row.iloc[0].get("plain_password")
                    if pwd and pd.notna(pwd):
                        return str(pwd)
        except Exception:
            pass
    return "sodecoton2024"  # fallback si le fichier n'existe pas
# ======================== SUITE DE L'APPLICATION (HEADER) ========================

st.markdown("""
<style>
    header[data-testid="stHeader"] { background-color: transparent !important; height: 2.5rem !important; box-shadow: none !important; }
    .fixed-header { position: fixed; top: 2.5rem; left: 0; right: 0; z-index: 999; background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); box-shadow: 0 2px 12px rgba(0,0,0,0.15); padding: 10px 2rem; display: flex; align-items: center; justify-content: space-between; height: 72px; }
    [data-testid="stSidebar"] { min-width: 280px !important; max-width: 300px !important; margin-top: 130px; }
    .main-content { margin-top: 130px; padding: 0 1rem; }
    section.main > div.block-container { padding-left: 2rem !important; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 8px; font-weight: 500; border: none; }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    h1, h2, h3 { color: #1b5e20; }
    .footer { position: fixed; bottom: 0; width: 100%; background-color: #1b5e20; color: white; text-align: center; padding: 8px; font-size: 13px; z-index: 999; }
    .info-box { background-color: #e8f5e9; border-left: 6px solid #2e7d32; padding: 12px 16px; border-radius: 8px; margin: 10px 0; }
    .warning-box { background-color: #fff8e1; border-left: 6px solid #f9a825; padding: 12px 16px; border-radius: 8px; margin: 10px 0; }
    .success-box { background-color: #e3f2fd; border-left: 6px solid #1565c0; padding: 12px 16px; border-radius: 8px; margin: 10px 0; }
    .error-box { background-color: #ffebee; border-left: 6px solid #c62828; padding: 12px 16px; border-radius: 8px; margin: 10px 0; }
    .kpi-card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid #2e7d32; text-align: center; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #1b5e20; }
    .kpi-label { font-size: 12px; color: #666; margin-top: 4px; }
    .badge-admin { background: #388e3c; color: white; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 6px; }
    .badge-viewer { background: #607d8b; color: white; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 6px; }
</style>
""", unsafe_allow_html=True)

logo_html = ""
if os.path.exists("LOGO SODEC.png"):
    with open("LOGO SODEC.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:44px; border-radius:4px;">'

role_badge = f'<span class="badge-admin">ADMIN</span>' if st.session_state.user_role == "admin" else f'<span class="badge-viewer">VIEWER</span>'

header_html = f"""
<div class="fixed-header">
    <div style="display:flex; align-items:center; gap:1rem;">
        {logo_html}
        <div>
            <div style="font-size:22px; font-weight:700; color:white;">SODECOTON</div>
            <div style="font-size:13px; color:#c8e6c9;">Module de Prédiction</div>
        </div>
    </div>
    <div style="color:#c8e6c9;">
        👤 <b style="color:white;">{st.session_state.username}</b>{role_badge}
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.markdown('<div class="main-content">', unsafe_allow_html=True)


# Chargement des données
DATA_FILE = "dataset_complet_2000_2026.csv"

def load_initial_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Fichier '{DATA_FILE}' introuvable.")
        st.stop()
    return pd.read_csv(DATA_FILE, encoding='utf-8-sig')

def preparer_donnees(df, pour_affichage=False):
    df = df.copy()
    target = "Rdt_region_kg_ha"
    num_cols = ['Pluie_Annuelle_mm', 'Credit_intrant_ha_FCFA', 'Temp_Moyenne_C', 'pH',
                'Matiere_Organique_pourcent', 'Capacite_Retention_mm', 'Date_Semis_jour',
                'Densite_plants_ha', 'Fertil_N_kg_ha', 'Fertil_P_kg_ha', 'Fertil_K_kg_ha',
                'Nb_Traitements', 'Superficie_ha', 'Latitude', 'Longitude']
    cat_cols = ['Région', 'Texture', 'Variete']
    for col in num_cols + cat_cols + [target]:
        if col not in df.columns:
            st.error(f"Colonne manquante : {col}")
            st.stop()
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([df[num_cols], df_encoded], axis=1)
    y = df[target]
    if pour_affichage:
        return X, y, df, num_cols, list(df_encoded.columns)
    return X, y, df

def choisir_cv(df_clean):
    if 'Année' in df_clean.columns:
        annees = df_clean['Année'].values
        n_annees = len(np.unique(annees))
        if n_annees >= 6:
            n_splits = min(5, n_annees - 1)
            return GroupKFold(n_splits=n_splits), annees, f"GroupKFold ({n_splits} plis, par année)"
    return KFold(n_splits=5, shuffle=True, random_state=42), None, "KFold (5 plis, shuffle)"

def train_model(df):
    X, y, df_orig = preparer_donnees(df, pour_affichage=False)
    if 'Année' in df_orig.columns:
        X = X.copy()
        X['Année'] = df_orig['Année'].values
    X_train = X.drop(columns=['Année']) if 'Année' in X.columns else X

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y)
    y_pred = model.predict(X_train)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = np.mean(np.abs((y - y_pred) / y.replace(0, np.nan))) * 100 if (y != 0).all() else np.nan

    cv_strategy, groupes, cv_nom = choisir_cv(df_orig)
    cv_r2, cv_rmse, cv_mae = [], [], []
    X_cv = X_train
    y_cv = y
    split_args = {'X': X_cv, 'y': y_cv}
    if groupes is not None:
        split_args['groups'] = groupes
    for train_idx, val_idx in cv_strategy.split(**split_args):
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        rf.fit(X_cv.iloc[train_idx], y_cv.iloc[train_idx])
        pred_val = rf.predict(X_cv.iloc[val_idx])
        cv_r2.append(r2_score(y_cv.iloc[val_idx], pred_val))
        cv_rmse.append(np.sqrt(mean_squared_error(y_cv.iloc[val_idx], pred_val)))
        cv_mae.append(mean_absolute_error(y_cv.iloc[val_idx], pred_val))

    biais = np.mean(y - y_pred)  # gardé pour compatibilité
    ref = df_orig.groupby('Région').agg({
        'Pluie_Annuelle_mm': 'median',
        'Credit_intrant_ha_FCFA': 'median',
        'Temp_Moyenne_C': 'median',
        'pH': 'median',
        'Matiere_Organique_pourcent': 'median',
        'Capacite_Retention_mm': 'median',
        'Date_Semis_jour': 'median',
        'Densite_plants_ha': 'median',
        'Fertil_N_kg_ha': 'median',
        'Fertil_P_kg_ha': 'median',
        'Fertil_K_kg_ha': 'median',
        'Nb_Traitements': 'median',
        'Superficie_ha': 'median',
        'Latitude': 'median',
        'Longitude': 'median',
        'Texture': lambda x: x.mode()[0] if not x.mode().empty else 'argileux',
        'Variete': lambda x: x.mode()[0] if not x.mode().empty else 'variete_A'
    }).reset_index()

    perf = {
        'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape,
        'cv_r2_mean': np.mean(cv_r2), 'cv_r2_std': np.std(cv_r2),
        'cv_rmse_mean': np.mean(cv_rmse), 'cv_rmse_std': np.std(cv_rmse),
        'cv_mae_mean': np.mean(cv_mae), 'n_estimators': 200,
        'ecart_surapp': round(r2 - np.mean(cv_r2), 3), 'cv_nom': cv_nom,
        'n_obs': len(df_orig), 'n_regions': df_orig['Région'].nunique(),
        'n_annees': df_orig['Année'].nunique() if 'Année' in df_orig.columns else 0,
    }
    return model, biais, ref, X_train.columns.tolist(), df_orig, X_train, y, perf

def comparer_modeles(df_clean, X_encoded, y):
    # simplifié : ne pas utiliser car RandomForest est déjà choisi, mais on garde pour compatibilité
    cv_strategy, groupes, cv_nom = choisir_cv(df_clean)
    resultats = []
    from sklearn.linear_model import RidgeCV, Ridge
    modeles = {
        "RidgeCV": RidgeCV(alphas=[0.1,1,10,50,100,500], cv=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }
    split_args = {'X': X_encoded, 'y': y}
    if groupes is not None:
        split_args['groups'] = groupes
    for nom, mod in modeles.items():
        cv_r2 = []
        for train_idx, val_idx in cv_strategy.split(**split_args):
            mod.fit(X_encoded.iloc[train_idx], y.iloc[train_idx])
            pred = mod.predict(X_encoded.iloc[val_idx])
            cv_r2.append(r2_score(y.iloc[val_idx], pred))
        resultats.append({"Modèle": nom, "R² CV moyen": round(np.mean(cv_r2),3)})
    return pd.DataFrame(resultats).sort_values("R² CV moyen", ascending=False), cv_nom

def predire_complet(region, pluie, credit, temp, ph, mat_org, cap_ret, date_semis, densite,
                    fertil_N, fertil_P, fertil_K, nb_trait, superficie, latitude, longitude,
                    texture, variete):
    try:
        input_dict = {
            "Pluie_Annuelle_mm": float(pluie), "Credit_intrant_ha_FCFA": float(credit),
            "Temp_Moyenne_C": float(temp), "pH": float(ph), "Matiere_Organique_pourcent": float(mat_org),
            "Capacite_Retention_mm": float(cap_ret), "Date_Semis_jour": float(date_semis),
            "Densite_plants_ha": float(densite), "Fertil_N_kg_ha": float(fertil_N),
            "Fertil_P_kg_ha": float(fertil_P), "Fertil_K_kg_ha": float(fertil_K),
            "Nb_Traitements": float(nb_trait), "Superficie_ha": float(superficie),
            "Latitude": float(latitude), "Longitude": float(longitude),
            "Région": str(region), "Texture": str(texture), "Variete": str(variete)
        }
        df_in = pd.DataFrame([input_dict])
        df_in = pd.get_dummies(df_in, columns=['Région','Texture','Variete'], drop_first=True)
        for col in colonnes_modele:
            if col not in df_in.columns:
                df_in[col] = 0
        df_in = df_in[colonnes_modele]
        result = float(modele.predict(df_in)[0]) + biais
        return round(max(0.0, result), 1)
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
        return 0.0

# Initialisation session
saved = load_model()
if saved is not None:
    st.session_state.modele = saved['modele']
    st.session_state.biais = saved['biais']
    st.session_state.ref_df = saved['ref_df']
    st.session_state.colonnes_modele = saved['colonnes_modele']
    st.session_state.data_df = saved['data_df']
    st.session_state.X_train = saved['X_train']
    st.session_state.y_train = saved['y_train']
    st.session_state.perf = saved['perf']
    st.session_state.df_original = saved['df_original']
else:
    df_initial = load_initial_data()
    (modele, biais, ref_df, colonnes_modele,
     data_df, X_train, y_train, perf) = train_model(df_initial)
    st.session_state.modele = modele
    st.session_state.biais = biais
    st.session_state.ref_df = ref_df
    st.session_state.colonnes_modele = colonnes_modele
    st.session_state.data_df = data_df
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.perf = perf
    st.session_state.df_original = df_initial
    save_model(modele, biais, ref_df, colonnes_modele, data_df, X_train, y_train, perf, df_initial)

modele = st.session_state.modele
biais = st.session_state.biais
ref_df = st.session_state.ref_df
colonnes_modele = st.session_state.colonnes_modele
data_df = st.session_state.data_df
X_train = st.session_state.X_train
y_train = st.session_state.y_train
perf = st.session_state.perf

def df_to_csv_bytes(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    return buffer.getvalue().encode('utf-8-sig')

# Navigation
PAGES = ["🏠 Accueil", "🔮 Prédiction par région",  "🎛️ Simulation", "📊 Performance du modèle", "🌍 Prédiction nationale", "⚙️ Administration", "❓ Aide"]
ADMIN_ONLY_PAGES = {"⚙️ Administration"}

st.sidebar.markdown("<h2 style='color:#1b5e20;'>🌿 Navigation</h2>", unsafe_allow_html=True)
if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]
pages_visibles = [p for p in PAGES if p not in ADMIN_ONLY_PAGES or st.session_state.user_role == "admin"]
def set_page(page_name):
    st.session_state.current_page = page_name
st.sidebar.radio("Choisissez une page :", pages_visibles,
                index=pages_visibles.index(st.session_state.current_page) if st.session_state.current_page in pages_visibles else 0,
                key="nav_menu", on_change=lambda: set_page(st.session_state.nav_menu))
st.sidebar.markdown("---")
badge_r2, label_r2 = r2_badge(perf.get('cv_r2_mean', 0))
st.sidebar.markdown(f"""
**Modele actuel (Random Forest)**

- R² CV : {perf.get('cv_r2_mean',0):.3f} {badge_r2}
- RMSE CV : {perf.get('cv_rmse_mean',0):.1f} kg/ha
""")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Données**\n- Observations : {perf.get('n_obs','—')}\n- Régions : {perf.get('n_regions','—')}\n- Années : {perf.get('n_annees','—')}")
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Déconnexion", width='stretch'):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.query_params.clear()
    st.rerun()

current_page = st.session_state.current_page

# ======================== PAGE ACCUEIL ========================
if current_page == "🏠 Accueil":
    st.header("Bienvenue dans SODECOTON Predict ")
    st.markdown(f"Connecté : **{st.session_state.username}** ({st.session_state.user_role})")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("R² CV", f"{perf['cv_r2_mean']:.3f}")
    with col2: st.metric("RMSE CV (kg/ha)", f"{perf['cv_rmse_mean']:.0f}")
    with col3: st.metric("Observations", perf['n_obs'])
    with col4: st.metric("Régions", perf['n_regions'])
    st.markdown("""
    **Modèle : Random Forest Regressor**  
    **Variables :** Pluie, Crédit, Température, pH, Matière organique, Capacité rétention, Date semis, Densité, Fertilisation N/P/K, Traitements, Superficie, Latitude, Longitude, Texture, Variété.  
    **Validation : GroupKFold par année.**  
    """)

# ======================== PRÉDICTION PAR RÉGION ========================
elif current_page == "🔮 Prédiction par région":
    st.header("Prédiction pour une région")
    st.markdown("<div class='info-box'>Saisissez les paramètres culturaux, climatiques et édaphiques pour estimer le rendement.</div>", unsafe_allow_html=True)
    with st.form("form_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("Région", sorted(data_df['Région'].unique()))
            pluie = st.number_input("Pluviométrie annuelle (mm)", 400, 1400, 800)
            credit = st.number_input("Crédit intrant/ha (FCFA)", 120000, 200000, 140000, step=1000)
            temp = st.number_input("Température moyenne (°C)", 24, 34, 28)
            ph = st.number_input("pH du sol", 4.5, 8.5, 6.5, step=0.1)
            mat_org = st.number_input("Matière organique (%)", 0.5, 3.0, 1.5, step=0.1)
            cap_ret = st.number_input("Capacité rétention (mm)", 50, 200, 100)
        with col2:
            date_semis = st.number_input("Date semis (jours après 1er janv.)", 120, 180, 140)
            densite = st.number_input("Densité (plants/ha)", 40000, 90000, 65000)
            fertil_N = st.number_input("Fertilisation N (kg/ha)", 30, 120, 60)
            fertil_P = st.number_input("Fertilisation P (kg/ha)", 20, 80, 40)
            fertil_K = st.number_input("Fertilisation K (kg/ha)", 20, 100, 50)
            nb_trait = st.number_input("Nombre de traitements", 1, 8, 3)
            superficie = st.number_input("Superficie (ha)", 5000, 120000, 30000)
            lat = st.number_input("Latitude", 8.0, 12.0, 9.5, step=0.1)
            lon = st.number_input("Longitude", 13.0, 15.0, 14.0, step=0.1)
            texture = st.selectbox("Texture", ["sablo-argileux", "limono-argileux", "argileux", "sablo-limoneux"])
            variete = st.selectbox("Variété", ["variete_A", "variete_B", "variete_C", "variete_D"])
        annee_p = st.number_input("Année de prédiction", 2000, 2050, datetime.date.today().year+1)
        submitted = st.form_submit_button("Lancer la prediction")
    if submitted:
        rdt = predire_complet(region, pluie, credit, temp, ph, mat_org, cap_ret, date_semis, densite,
                              fertil_N, fertil_P, fertil_K, nb_trait, superficie, lat, lon,
                              texture, variete)
        rmse_cv = perf['cv_rmse_mean']
        prod_est = superficie * rdt / 1000
        colA, colB, colC = st.columns(3)
        colA.metric("Rendement estimé (kg/ha)", format_fr(rdt), delta=f"±{format_fr(rmse_cv)}")
        colB.metric("Production estimée (T)", format_fr(prod_est))
        colC.metric("Intervalle (kg/ha)", f"{format_fr(max(0, rdt-rmse_cv))} – {format_fr(rdt+rmse_cv)}")
        fig, ax = plt.subplots(figsize=(8,2))
        ax.barh([0], [rdt], color='#2e7d32', height=0.4, label=f"Prédit : {format_fr(rdt)} kg/ha")
        ax.barh([0], [rdt+rmse_cv], color='#a5d6a7', height=0.2, left=0, alpha=0.5, label=f"Borne haute : {format_fr(rdt+rmse_cv)}")
        ax.set_xlim(0, max(rdt+rmse_cv+100, 1200))
        ax.set_yticks([])
        ax.set_xlabel("Rendement (kg/ha)")
        ax.set_title(f"Prédiction — {region} ({annee_p})")
        ax.legend(loc='lower right')
        st.pyplot(fig)
        result_df = pd.DataFrame([{
            "Région": region, "Année": annee_p, "Crédit_FCFA": credit,
            "Rendement_prédit_kg_ha": rdt,
            "Borne_inf": max(0, rdt-rmse_cv), "Borne_sup": rdt+rmse_cv,
            "Production_estimée_T": round(prod_est,1)
        }])
        st.download_button("Télécharger la prédiction (CSV)", data=df_to_csv_bytes(result_df),
                           file_name=f"prediction_{region}_{annee_p}.csv", mime="text/csv")

# ======================== PRÉDICTION NATIONALE ========================
elif current_page == "🌍 Prédiction nationale":
    st.header("Prédiction nationale")
    st.markdown("<div class='info-box'>Ajustez les paramètres pour chaque région, puis calculez le rendement national et la production totale estimée.</div>", unsafe_allow_html=True)

    # Initialisation des paramètres régionaux dans la session
    if "param_regions" not in st.session_state:
        st.session_state.param_regions = {}

    regions = sorted(data_df["Région"].unique())
    # Récupérer les superficies de la dernière année
    derniere_annee = data_df["Année"].max()
    df_last = data_df[data_df["Année"] == derniere_annee]
    default_superficies = df_last.set_index("Région")["Superficie_ha"].to_dict()

    # Récupérer les autres valeurs par défaut depuis ref_df
    default_values = ref_df.set_index("Région").to_dict(orient="index")

    # Initialiser ou mettre à jour chaque région
    for reg in regions:
        if reg not in st.session_state.param_regions:
            st.session_state.param_regions[reg] = {
                "Pluie_Annuelle_mm": default_values[reg]["Pluie_Annuelle_mm"],
                "Credit_intrant_ha_FCFA": default_values[reg]["Credit_intrant_ha_FCFA"],
                "Temp_Moyenne_C": default_values[reg]["Temp_Moyenne_C"],
                "pH": default_values[reg]["pH"],
                "Matiere_Organique_pourcent": default_values[reg]["Matiere_Organique_pourcent"],
                "Capacite_Retention_mm": default_values[reg]["Capacite_Retention_mm"],
                "Date_Semis_jour": default_values[reg]["Date_Semis_jour"],
                "Densite_plants_ha": default_values[reg]["Densite_plants_ha"],
                "Fertil_N_kg_ha": default_values[reg]["Fertil_N_kg_ha"],
                "Fertil_P_kg_ha": default_values[reg]["Fertil_P_kg_ha"],
                "Fertil_K_kg_ha": default_values[reg]["Fertil_K_kg_ha"],
                "Nb_Traitements": default_values[reg]["Nb_Traitements"],
                "Superficie_ha": default_superficies.get(reg, 30000),
                "Latitude": default_values[reg]["Latitude"],
                "Longitude": default_values[reg]["Longitude"],
                "Texture": default_values[reg]["Texture"],
                "Variete": default_values[reg]["Variete"]
            }
        else:
            # Mise à jour des clés manquantes (pour les sessions existantes)
            if "Superficie_ha" not in st.session_state.param_regions[reg]:
                st.session_state.param_regions[reg]["Superficie_ha"] = default_superficies.get(reg, 30000)

    # Sélection de la région à modifier
    region_selected = st.selectbox("Choisissez une région à modifier", regions)

    with st.expander(f"Paramètres de {region_selected}", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            pluie = st.slider("Pluviométrie (mm)", 400, 1400, int(st.session_state.param_regions[region_selected]["Pluie_Annuelle_mm"]), key=f"pluie_{region_selected}")
            credit = st.slider("Crédit intrant (FCFA)", 120000, 200000, int(st.session_state.param_regions[region_selected]["Credit_intrant_ha_FCFA"]), step=1000, key=f"credit_{region_selected}")
            temp = st.slider("Température (°C)", 24, 34, int(round(st.session_state.param_regions[region_selected]["Temp_Moyenne_C"])), key=f"temp_{region_selected}")
        with col2:
            ph = st.slider("pH", 4.5, 8.5, float(st.session_state.param_regions[region_selected]["pH"]), step=0.1, key=f"ph_{region_selected}")
            mo = st.slider("Matière organique (%)", 0.5, 3.0, float(st.session_state.param_regions[region_selected]["Matiere_Organique_pourcent"]), step=0.1, key=f"mo_{region_selected}")
            cap = st.slider("Capacité rétention (mm)", 50, 200, int(st.session_state.param_regions[region_selected]["Capacite_Retention_mm"]), key=f"cap_{region_selected}")
        with col3:
            date_semis = st.slider("Date semis (jours)", 120, 180, int(st.session_state.param_regions[region_selected]["Date_Semis_jour"]), key=f"date_{region_selected}")
            densite = st.slider("Densité (plants/ha)", 40000, 90000, int(st.session_state.param_regions[region_selected]["Densite_plants_ha"]), step=1000, key=f"dens_{region_selected}")
            trait = st.slider("Nb traitements", 1, 8, int(st.session_state.param_regions[region_selected]["Nb_Traitements"]), key=f"trait_{region_selected}")
        col4, col5, col6 = st.columns(3)
        with col4:
            n = st.slider("N (kg/ha)", 30, 120, int(st.session_state.param_regions[region_selected]["Fertil_N_kg_ha"]), key=f"n_{region_selected}")
        with col5:
            p = st.slider("P (kg/ha)", 20, 80, int(st.session_state.param_regions[region_selected]["Fertil_P_kg_ha"]), key=f"p_{region_selected}")
        with col6:
            k = st.slider("K (kg/ha)", 20, 100, int(st.session_state.param_regions[region_selected]["Fertil_K_kg_ha"]), key=f"k_{region_selected}")

        # Curseur spécifique pour la superficie
        superficie_defaut = st.session_state.param_regions[region_selected]["Superficie_ha"]
        superficie = st.slider("Superficie (ha)", 5000, 120000, int(superficie_defaut), step=1000, key=f"sup_{region_selected}")

        if st.button(f"Appliquer pour {region_selected}"):
            st.session_state.param_regions[region_selected]["Pluie_Annuelle_mm"] = pluie
            st.session_state.param_regions[region_selected]["Credit_intrant_ha_FCFA"] = credit
            st.session_state.param_regions[region_selected]["Temp_Moyenne_C"] = temp
            st.session_state.param_regions[region_selected]["pH"] = ph
            st.session_state.param_regions[region_selected]["Matiere_Organique_pourcent"] = mo
            st.session_state.param_regions[region_selected]["Capacite_Retention_mm"] = cap
            st.session_state.param_regions[region_selected]["Date_Semis_jour"] = date_semis
            st.session_state.param_regions[region_selected]["Densite_plants_ha"] = densite
            st.session_state.param_regions[region_selected]["Fertil_N_kg_ha"] = n
            st.session_state.param_regions[region_selected]["Fertil_P_kg_ha"] = p
            st.session_state.param_regions[region_selected]["Fertil_K_kg_ha"] = k
            st.session_state.param_regions[region_selected]["Nb_Traitements"] = trait
            st.session_state.param_regions[region_selected]["Superficie_ha"] = superficie
            st.success(f"✅ Paramètres de {region_selected} mis à jour")

   

    # Bouton de calcul
    if st.button("🌍 Calculer le rendement national", use_container_width=True):
        resultats = []
        preds = []
        surf_vals = []

        for reg in regions:
            params = st.session_state.param_regions[reg]
            # La superficie est maintenant une variable d'entrée pour la prédiction
            rdt = predire_complet(
                reg,
                params["Pluie_Annuelle_mm"],
                params["Credit_intrant_ha_FCFA"],
                params["Temp_Moyenne_C"],
                params["pH"],
                params["Matiere_Organique_pourcent"],
                params["Capacite_Retention_mm"],
                params["Date_Semis_jour"],
                params["Densite_plants_ha"],
                params["Fertil_N_kg_ha"],
                params["Fertil_P_kg_ha"],
                params["Fertil_K_kg_ha"],
                params["Nb_Traitements"],
                params["Superficie_ha"],  # <-- on utilise la superficie saisie
                params["Latitude"],
                params["Longitude"],
                params["Texture"],
                params["Variete"]
            )
            preds.append(rdt)
            surf_vals.append(params["Superficie_ha"])
            resultats.append({"Région": reg, "Rendement (kg/ha)": rdt, "Superficie (ha)": params["Superficie_ha"]})

        total_surf = sum(surf_vals)
        national_kg_ha = sum([p * s for p, s in zip(preds, surf_vals)]) / total_surf if total_surf > 0 else 0
        production_totale_tonnes = (national_kg_ha * total_surf) / 1000

        # Stockage pour export
        st.session_state.df_res = pd.DataFrame(resultats)
        st.session_state.national_kg_ha = national_kg_ha
        st.session_state.production_totale_tonnes = production_totale_tonnes
        st.session_state.total_surf = total_surf

        st.subheader("📊 Résultat national")
        colA, colB, colC = st.columns(3)
        colA.metric("Rendement national (kg/ha)", f"{national_kg_ha:.1f}")
        colB.metric("Production totale estimée (tonnes)", f"{production_totale_tonnes:,.0f}".replace(",", " "))
        colC.metric("Superficie totale", f"{total_surf:,.0f} ha".replace(",", " "))

       
        # Export CSV
        csv_data = st.session_state.df_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📊 Télécharger les données (CSV)",
            csv_data,
            file_name="resultats_nationaux.csv",
            mime="text/csv"
        )
# ======================== SIMULATION ========================
elif current_page == "🎛️ Simulation":
    st.header("Simulateur avancé")
    st.markdown("<div class='info-box'>Simulez l'impact de chaque variable sur le rendement et comparez toutes les régions.</div>", unsafe_allow_html=True)
    region = st.selectbox("Région", sorted(data_df['Région'].unique()), key="sim_region")
    ref_region = ref_df[ref_df['Région'] == region].iloc[0]
    def safe_val(col, default):
        return ref_region[col] if col in ref_region else default
    col1, col2, col3 = st.columns(3)
    with col1:
        pluie = st.slider("Pluviométrie (mm)", 400, 1400, int(safe_val('Pluie_Annuelle_mm', 800)), key="sim_pluie")
        credit = st.slider("Crédit intrant (FCFA)", 120000, 200000, int(safe_val('Credit_intrant_ha_FCFA', 140000)), step=1000, key="sim_credit")
        temp = st.slider("Température (°C)", 24, 34, int(round(safe_val('Temp_Moyenne_C', 28))), key="sim_temp")
        ph = st.slider("pH du sol", 4.5, 8.5, float(safe_val('pH', 6.5)), step=0.1, key="sim_ph")
    with col2:
        fertil_N = st.slider("Fertilisation N (kg/ha)", 30, 120, int(safe_val('Fertil_N_kg_ha', 60)), key="sim_n")
        fertil_P = st.slider("Fertilisation P (kg/ha)", 20, 80, int(safe_val('Fertil_P_kg_ha', 40)), key="sim_p")
        fertil_K = st.slider("Fertilisation K (kg/ha)", 20, 100, int(safe_val('Fertil_K_kg_ha', 50)), key="sim_k")
        nb_trait = st.slider("Nombre de traitements", 1, 8, int(safe_val('Nb_Traitements', 3)), key="sim_trait")
    with col3:
        date_semis = st.slider("Date de semis (jours)", 120, 180, int(safe_val('Date_Semis_jour', 140)), key="sim_semis")
        densite = st.slider("Densité (plants/ha)", 40000, 90000, int(safe_val('Densite_plants_ha', 65000)), step=1000, key="sim_densite")
        superficie = st.slider("Superficie (ha)", 5000, 120000, int(safe_val('Superficie_ha', 30000)), step=1000, key="sim_superficie")
    mat_org = safe_val('Matiere_Organique_pourcent', 1.5)
    cap_ret = safe_val('Capacite_Retention_mm', 100)
    lat = safe_val('Latitude', 9.5)
    lon = safe_val('Longitude', 14.0)
    texture = safe_val('Texture', 'argileux')
    variete = safe_val('Variete', 'variete_A')
    rdt = predire_complet(region, pluie, credit, temp, ph, mat_org, cap_ret, date_semis, densite,
                          fertil_N, fertil_P, fertil_K, nb_trait, superficie, lat, lon,
                          texture, variete)
    prod_est = superficie * rdt / 1000
    st.info(f"Rendement estimé pour **{region}** : **{format_fr(rdt)} kg/ha** → production estimée : **{format_fr(prod_est)} tonnes**")
   

# ======================== PERFORMANCE DU MODÈLE ========================
elif current_page == "📊 Performance du modèle":
    st.header("Performance du modèle ")
    badge, label = r2_badge(perf['cv_r2_mean'])
    st.markdown(f"""<div class='success-box'>Qualité : {badge} <b>{label}</b> (R² CV = {perf['cv_r2_mean']:.3f})</div>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² CV moyen", f"{perf['cv_r2_mean']:.3f} ±{perf['cv_r2_std']:.3f}")
    col2.metric("RMSE CV (kg/ha)", f"{perf['cv_rmse_mean']:.1f} ±{perf['cv_rmse_std']:.1f}")
    col3.metric("MAE CV (kg/ha)", f"{perf['cv_mae_mean']:.1f}")
    col4.metric("Écart surapprentissage", f"{perf['ecart_surapp']:.3f}")
    st.subheader("Prédictions vs Valeurs réelles")
    y_pred_train = modele.predict(X_train)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(y_train, y_pred_train + biais, alpha=0.6, color='#2e7d32')
    lims = [min(y_train.min(), (y_pred_train+biais).min())-50, max(y_train.max(), (y_pred_train+biais).max())+50]
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel("Rendement réel (kg/ha)"); ax.set_ylabel("Rendement prédit (kg/ha)")
    ax.set_title(f"Prédictions vs Réels - R² = {perf['r2']:.3f}")
    st.pyplot(fig)
   
    if 'Année' in data_df.columns:
        st.subheader("Rendement réel vs prédit par année")
        y_pred_all = modele.predict(X_train)
        df_plot = data_df.copy()
        df_plot['Prédit'] = y_pred_all + biais
        df_plot_agg = df_plot.groupby('Année').agg(Réel=('Rdt_region_kg_ha','mean'), Prédit=('Prédit','mean')).reset_index()
        fig4, ax4 = plt.subplots(figsize=(10,4))
        ax4.plot(df_plot_agg['Année'], df_plot_agg['Réel'], 'o-', label='Réel', color='#1b5e20')
        ax4.plot(df_plot_agg['Année'], df_plot_agg['Prédit'], 's--', label='Prédit', color='#ff9800')
        ax4.fill_between(df_plot_agg['Année'], df_plot_agg['Prédit'] - perf['cv_rmse_mean'],
                         df_plot_agg['Prédit'] + perf['cv_rmse_mean'], alpha=0.15, color='#ff9800', label='±RMSE CV')
        ax4.set_xlabel("Année"); ax4.set_ylabel("Rendement moyen (kg/ha)")
        ax4.set_title("Rendement moyen national : réel vs prédit")
        ax4.legend()
        st.pyplot(fig4)
# ======================== ADMINISTRATION ========================
elif current_page == "⚙️ Administration":
    if st.session_state.user_role != "admin":
        st.error("Accès réservé aux administrateurs.")
        st.stop()
    st.header("Administration")
    tab_data, tab_model, tab_users = st.tabs(["Données & Import", "Modèle", "Utilisateurs"])

    # ========== ONGLET DONNÉES & IMPORT ==========
    with tab_data:
        st.subheader("Structure des données")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", X_train.shape[0])
        col2.metric("Variables", len(colonnes_modele))
        has_annee = 'Année' in data_df.columns
        n_annees = data_df['Année'].nunique() if has_annee else 0
        col3.metric("Années distinctes", n_annees if has_annee else "Absent")

        st.markdown("---")
        st.subheader("Importer un dataset")

        # ========== INDICATION DU FORMAT ATTENDU ==========
        with st.expander("📋 Format attendu du fichier CSV", expanded=False):
            st.markdown("""
            Le fichier doit être au format **CSV** (séparateur virgule, encodage UTF-8) et contenir les colonnes suivantes :

            | Colonne | Type | Description |
            |---------|------|-------------|
            | `Région` | Texte | Nom de la région |
            | `Année` | Entier | Année de la campagne |
            | `Pluie_Annuelle_mm` | Numérique | Cumul pluviométrique (mm) |
            | `Credit_intrant_ha_FCFA` | Numérique | Crédit intrant (FCFA/ha) |
            | `Rdt_region_kg_ha` | Numérique | Rendement (kg/ha) – **variable cible** |
            | `Superficie_ha` | Numérique | Superficie cultivée (ha) |
            | `Temp_Moyenne_C` | Numérique | Température moyenne (°C) |
            | `pH` | Numérique | pH du sol |
            | `Matiere_Organique_pourcent` | Numérique | Matière organique (%) |
            | `Capacite_Retention_mm` | Numérique | Capacité de rétention (mm) |
            | `Texture` | Texte | Texture du sol |
            | `Date_Semis_jour` | Numérique | Date de semis (jours après 1er janv.) |
            | `Densite_plants_ha` | Numérique | Densité de plantation (plants/ha) |
            | `Fertil_N_kg_ha` | Numérique | Engrais azoté (kg/ha) |
            | `Fertil_P_kg_ha` | Numérique | Engrais phosphoré (kg/ha) |
            | `Fertil_K_kg_ha` | Numérique | Engrais potassique (kg/ha) |
            | `Nb_Traitements` | Numérique | Nombre de traitements phytosanitaires |
            | `Variete` | Texte | Variété de semence |
            | `Latitude` | Numérique | Latitude de la région |
            | `Longitude` | Numérique | Longitude de la région |

            ⚠️ **Toute colonne manquante ou de type incorrect bloquera l'importation.**
            """)

        uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                required_cols = [
                    'Région', 'Année', 'Pluie_Annuelle_mm', 'Credit_intrant_ha_FCFA',
                    'Rdt_region_kg_ha', 'Superficie_ha', 'Temp_Moyenne_C', 'pH',
                    'Matiere_Organique_pourcent', 'Capacite_Retention_mm', 'Texture',
                    'Date_Semis_jour', 'Densite_plants_ha', 'Fertil_N_kg_ha',
                    'Fertil_P_kg_ha', 'Fertil_K_kg_ha', 'Nb_Traitements',
                    'Variete', 'Latitude', 'Longitude'
                ]
                missing = [col for col in required_cols if col not in new_data.columns]
                if missing:
                    st.error(f"❌ Colonnes manquantes : {', '.join(missing)}")
                    st.info("📋 Consultez l'expandeur ci-dessus pour le format attendu.")
                else:
                    st.success(f"✅ {len(new_data)} lignes chargées, format valide.")
                    if st.button("Remplacer et réentraîner"):
                        with st.spinner("Réentraînement..."):
                            m, b, ref, cols, d, Xt, yt, perf_new = train_model(new_data)
                            st.session_state.update({
                                'modele': m,
                                'biais': b,
                                'ref_df': ref,
                                'colonnes_modele': cols,
                                'data_df': d,
                                'X_train': Xt,
                                'y_train': yt,
                                'perf': perf_new,
                                'df_original': new_data
                            })
                            save_model(m, b, ref, cols, d, Xt, yt, perf_new, new_data)
                            st.success(f"✅ Modèle réentraîné - R² CV = {perf_new['cv_r2_mean']:.3f}")
                            st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur de lecture du fichier : {e}")
                st.info("📋 Assurez-vous que le fichier est un CSV valide (séparateur virgule, encodage UTF-8).")

        st.markdown("---")
        st.subheader("Exporter le dataset actuel")
        st.download_button(
            "Télécharger CSV",
            data=df_to_csv_bytes(st.session_state.df_original),
            file_name=f"dataset_{datetime.date.today()}.csv"
        )

    # ========== ONGLET MODÈLE ==========
    with tab_model:
        st.subheader("Réentraîner le modèle avec les données actuelles")
        if st.button("Réentraîner", width='stretch'):
            with st.spinner("Réentraînement..."):
                m, b, ref, cols, d, Xt, yt, perf_new = train_model(st.session_state.df_original)
                st.session_state.update({
                    'modele': m,
                    'biais': b,
                    'ref_df': ref,
                    'colonnes_modele': cols,
                    'data_df': d,
                    'X_train': Xt,
                    'y_train': yt,
                    'perf': perf_new
                })
                save_model(m, b, ref, cols, d, Xt, yt, perf_new, st.session_state.df_original)
                st.success(f"✅ Modèle réentraîné - R² CV = {perf_new['cv_r2_mean']:.3f}")
                st.rerun()
        st.subheader("Résumé du modèle")
        st.write(f"Observations : {X_train.shape[0]}")
        st.write(f"Variables : {len(colonnes_modele)}")
        st.write(f"Stratégie CV : {perf.get('cv_nom','N/A')}")
        st.write(f"R² CV : {perf.get('cv_r2_mean',0):.3f}")
        st.write(f"RMSE CV : {perf.get('cv_rmse_mean',0):.1f} kg/ha")

    # ========== ONGLET UTILISATEURS ==========
    with tab_users:
        st.subheader("Liste des utilisateurs")
        users = load_users()
        st.dataframe(users[['username','role','created_at']], use_container_width=True)
        st.markdown("---")
        st.subheader("Ajouter un utilisateur")
        col1, col2, col3 = st.columns(3)
        new_u = col1.text_input("Nom")
        new_p = col2.text_input("Mot de passe", type="password")
        new_r = col3.selectbox("Rôle", ["viewer","admin"])
        if st.button("Ajouter"):
            success, msg = add_user(new_u, new_p, new_r, load_users())
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        st.markdown("---")
        st.subheader("Supprimer un utilisateur")
        users_list = [u for u in load_users()["username"].tolist() if u != "admin"]
        if users_list:
            del_user = st.selectbox("Utilisateur", users_list)
            if st.button("Supprimer"):
                success, msg = delete_user(del_user, load_users())
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("Aucun utilisateur supprimable (admin protégé)")

# ======================== AIDE ========================
elif current_page == "❓ Aide":
    st.header("Aide - Guide d'utilisation")
    
    # --- Introduction ---
    st.markdown("""
    Bienvenue dans le module de **prédiction du rendement cotonnier** de la SODECOTON.  
    Cet outil vous permet d'estimer le rendement en coton graine (kg/ha) à l'échelle régionale, en vous appuyant sur un modèle de **Machine Learning** (Random Forest) entraîné sur 27 années de données historiques.
    
    ---
    """)

    # --- Fonctionnalités générales ---
    st.subheader("📌 Fonctionnalités disponibles")
    st.markdown("""
    - **Prédiction par région** : Saisissez les paramètres culturaux (pluviométrie, crédit, fertilisation, etc.) pour obtenir une estimation du rendement avec un intervalle de confiance.
    - **Simulateur avancé** : Faites varier dynamiquement les variables clés (pluie, crédit, fertilisation, date de semis, etc.) et visualisez en temps réel l’impact sur le rendement.
    - **Performance du modèle** : Consultez les métriques de validation (R², RMSE, MAE), les graphiques prédictions vs réalisé, et l’importance des variables.
    - **Analyse exploratoire (EDA)** : Explorez les distributions, les corrélations, et les évolutions temporelles des données.
    - **Administration** (réservé aux admins) : Gérez les utilisateurs (ajout/suppression), importez un nouveau jeu de données et réentraînez le modèle.
    """)

    # --- Aide sur les saisies ---
    st.subheader("📋 Variables à renseigner")
    st.markdown("""
    Les principales variables demandées sont :
    - **Pluviométrie annuelle** (mm) – cumul des précipitations sur la saison.
    - **Crédit intrant** (FCFA/ha) – montant investi en semences, engrais, pesticides.
    - **Température moyenne** (°C) – moyenne de la saison.
    - **pH du sol** et **matière organique** (%) – caractéristiques édaphiques.
    - **Date de semis** (en jours après le 1er janvier).
    - **Densité de plantation** (plants/ha).
    - **Fertilisation** (N, P, K en kg/ha).
    - **Nombre de traitements phytosanitaires**.
    - **Superficie** (ha), **latitude**, **longitude**, **texture** et **variété**.
    
    Tous les champs sont fournis avec des valeurs par défaut réalistes. Vous pouvez les ajuster à l’aide de curseurs ou de menus déroulants.
    """)

    # --- Rappel du modèle ---
    st.subheader("🧠 Modèle prédictif")
    st.markdown("""
    - **Algorithme** : Random Forest Regressor (forêt aléatoire).
    - **Validation** : cross-validation temporelle (GroupKFold) sur 5 plis, respectant l’ordre chronologique des années.
    - **Performances** : R² = 0,792 ; RMSE = 45,4 kg/ha ; MAE = 35,8 kg/ha.
    - **Interprétabilité** : le modèle fournit l’importance des variables, et des graphiques SHAP expliquent chaque prédiction.
    """)

    # --- Section Admin (uniquement pour l'admin) ---
    if st.session_state.user_role == "admin":
        st.markdown("---")
        st.subheader("🔐 Administrateur")
        st.markdown("En tant qu'administrateur, vous avez accès à la page **Administration** où vous pouvez :")
        st.markdown("""
        - Créer de nouveaux comptes utilisateurs (avec rôle *viewer* ou *admin*).
        - Supprimer des comptes (sauf le vôtre).
        - Importer un nouveau jeu de données au format CSV.
        - Réentraîner le modèle sur les nouvelles données.
        """)
        # Récupération du mot de passe actuel
        admin_password = get_admin_password()
        st.markdown(f"""
        **Informations de connexion par défaut :**  
        - Identifiant : `admin`  
        - Mot de passe actuel : `{admin_password}`
        
        *Ce mot de passe est mis à jour automatiquement si vous le modifiez via la réinitialisation ou la page d'administration.*
        """)
    else:
        st.markdown("---")
        st.info("ℹ️ Pour toute question relative à votre compte (oubli de mot de passe, modification des droits, etc.), veuillez contacter l'administrateur.")

    # --- Footer de la page ---
    st.markdown("---")
    st.caption("© 2026 SODECOTON — Module de prédiction de rendement cotonnier")
    
    # ========== FOOTER (s'applique à toutes les pages) ==========
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(f"<div class='footer'>© 2026 SODECOTON — Tous droits réservés | 👤 {st.session_state.username} ({st.session_state.user_role})</div>", unsafe_allow_html=True)
