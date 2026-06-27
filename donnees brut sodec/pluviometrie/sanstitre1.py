# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:35:05 2026

@author: THIS PC
"""

import pandas as pd
import numpy as np
import glob
import os
import re

# ------------------------------------------------------------
# 1. DÉTECTION DU DOSSIER PRINCIPAL "pluviometrie"
# ------------------------------------------------------------
bureau = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(bureau):
    bureau = os.path.join(os.path.expanduser("~"), "Bureau")

dossier_principal = os.path.join(bureau, "pluviometrie")

if not os.path.exists(dossier_principal):
    print(f"⚠️ Dossier '{dossier_principal}' introuvable.")
    chemin_manuel = input("Entrez le chemin complet du dossier 'pluviometrie' : ").strip()
    if os.path.exists(chemin_manuel):
        dossier_principal = chemin_manuel
    else:
        raise FileNotFoundError("Dossier non trouvé.")

print(f"✅ Dossier principal : {dossier_principal}")

# ------------------------------------------------------------
# 2. SOUS-DOSSIERS ATTENDUS
# ------------------------------------------------------------
dossier_annuel = os.path.join(dossier_principal, "annuel")
dossier_mensuel = os.path.join(dossier_principal, "mensuel")
dossier_journalier = os.path.join(dossier_principal, "journalier")

# Créer les dossiers s'ils n'existent pas
for d in [dossier_annuel, dossier_mensuel, dossier_journalier]:
    os.makedirs(d, exist_ok=True)

print("\n📂 Organisation des dossiers :")
print(f"   Annuel    : {dossier_annuel}")
print(f"   Mensuel   : {dossier_mensuel}")
print(f"   Journalier: {dossier_journalier}")

# ------------------------------------------------------------
# 3. CORRESPONDANCE DES STATIONS
# ------------------------------------------------------------
CORRESP_STATIONS = {
    'MAROUA': 'Maroua',
    'KAELE': 'Kaélé',
    'TCHATIBALI': 'Tchatibali',
    'TBLI': 'Tchatibali',
    'GUIDER': 'Guider',
    'GAROUA': 'Garoua',
    'NGONG': 'Ngong',
    'MAYO-GALKE': 'Mayo Galké',
    'M-GALKE': 'Mayo Galké',
    'POLI': 'Poli',
    'TOUBORO': 'Touboro',
    'TOUBOURO': 'Touboro',
    'HOME': 'Homé'
}

# ------------------------------------------------------------
# 4. FONCTIONS DE LECTURE
# ------------------------------------------------------------
def lire_fichier_annuel(chemin):
    """Lit un fichier annuel et retourne un DataFrame long."""
    df = pd.read_excel(chemin, sheet_name='Feuil1')
    df = df.rename(columns={df.columns[0]: 'Année'})
    df = df[df['Année'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]
    df['Année'] = df['Année'].astype(int)
    df_long = df.melt(id_vars='Année', var_name='Station', value_name='Pluie_Annuelle_mm')
    df_long['Station'] = df_long['Station'].str.strip()
    df_long = df_long.dropna(subset=['Pluie_Annuelle_mm'])
    return df_long

def extraire_mois_annee(nom_feuille):
    """Extrait le mois et l'année du nom d'une feuille Excel."""
    mois_map = {
        'JANVIER':1, 'FEVRIER':2, 'MARS':3, 'AVRIL':4, 'MAI':5, 'JUIN':6,
        'JUILLET':7, 'AOUT':7, 'AOÛT':8, 'SEPTEMBRE':9, 'OCTOBRE':10,
        'NOVEMBRE':11, 'DECEMBRE':12
    }
    pattern = r'(?P<mois>[A-ZÀÛ]+)\s*(?P<annee>\d{4})'
    match = re.search(pattern, nom_feuille.upper())
    if match:
        mois_str = match.group('mois')
        annee = int(match.group('annee'))
        for nom, num in mois_map.items():
            if nom in mois_str:
                return num, annee
    return None, None

def lire_fichier_mensuel(chemin):
    """
    Lit un fichier mensuel (contenant des feuilles par mois avec cumuls mensuels)
    Retourne un DataFrame avec les cumuls mensuels par station.
    """
    xl = pd.ExcelFile(chemin)
    liste_lignes = []
    for sheet in xl.sheet_names:
        if sheet in ['IMPRIME', 'Feuille12', 'Feuille13']:
            continue
        mois, annee = extraire_mois_annee(sheet)
        if mois is None or annee is None:
            continue
        
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        # Les totaux mensuels sont en colonne AG (index 32) pour les lignes 8 à 17
        for row_idx in range(7, 17):
            nom_station_brut = str(df.iloc[row_idx, 0]).strip().upper()
            if nom_station_brut not in CORRESP_STATIONS:
                continue
            station_std = CORRESP_STATIONS[nom_station_brut]
            val = df.iloc[row_idx, 32]  # colonne AG
            if pd.isna(val) or val == '':
                continue
            try:
                pluie = float(val)
            except:
                continue
            liste_lignes.append({
                'Année': annee,
                'Mois': mois,
                'Station': station_std,
                'Pluie_Mensuelle_mm': pluie
            })
    if not liste_lignes:
        return pd.DataFrame()
    return pd.DataFrame(liste_lignes)

def lire_fichier_journalier(chemin):
    """
    Lit un fichier journalier (multi-feuilles avec jours du mois)
    Retourne un DataFrame journalier.
    """
    xl = pd.ExcelFile(chemin)
    liste_lignes = []
    for sheet in xl.sheet_names:
        if sheet in ['IMPRIME', 'Feuille12', 'Feuille13']:
            continue
        mois, annee = extraire_mois_annee(sheet)
        if mois is None or annee is None:
            continue
        
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        jours = df.iloc[6, 1:32].values  # colonnes B à AF
        jours_vals = []
        for j in jours:
            try:
                jours_vals.append(int(j))
            except:
                jours_vals.append(np.nan)
        
        for row_idx in range(7, 17):
            nom_station_brut = str(df.iloc[row_idx, 0]).strip().upper()
            if nom_station_brut not in CORRESP_STATIONS:
                continue
            station_std = CORRESP_STATIONS[nom_station_brut]
            for col_idx, jour in enumerate(jours_vals, start=1):
                if pd.isna(jour):
                    continue
                val = df.iloc[row_idx, col_idx]
                if pd.isna(val) or val == '':
                    continue
                try:
                    pluie = float(val)
                except:
                    continue
                date = pd.Timestamp(year=annee, month=mois, day=int(jour))
                liste_lignes.append({
                    'Date': date,
                    'Station': station_std,
                    'Pluie_mm': pluie
                })
    if not liste_lignes:
        return pd.DataFrame()
    return pd.DataFrame(liste_lignes)

# ------------------------------------------------------------
# 5. TRAITEMENT DES TROIS DOSSIERS
# ------------------------------------------------------------
print("\n🔍 Traitement des fichiers...")

df_annuel = pd.DataFrame()
df_mensuel = pd.DataFrame()
df_journalier = pd.DataFrame()

# --- Dossier Annuel ---
fichiers_annuel = glob.glob(os.path.join(dossier_annuel, "*.xlsx")) + glob.glob(os.path.join(dossier_annuel, "*.xls"))
print(f"\n📁 Dossier 'annuel' : {len(fichiers_annuel)} fichier(s)")
for fichier in fichiers_annuel:
    print(f"   -> {os.path.basename(fichier)}")
    df_temp = lire_fichier_annuel(fichier)
    df_annuel = pd.concat([df_annuel, df_temp], ignore_index=True)

# --- Dossier Mensuel ---
fichiers_mensuel = glob.glob(os.path.join(dossier_mensuel, "*.xlsx")) + glob.glob(os.path.join(dossier_mensuel, "*.xls"))
print(f"\n📁 Dossier 'mensuel' : {len(fichiers_mensuel)} fichier(s)")
for fichier in fichiers_mensuel:
    print(f"   -> {os.path.basename(fichier)}")
    df_temp = lire_fichier_mensuel(fichier)
    df_mensuel = pd.concat([df_mensuel, df_temp], ignore_index=True)

# --- Dossier Journalier ---
fichiers_journalier = glob.glob(os.path.join(dossier_journalier, "*.xlsx")) + glob.glob(os.path.join(dossier_journalier, "*.xls"))
print(f"\n📁 Dossier 'journalier' : {len(fichiers_journalier)} fichier(s)")
for fichier in fichiers_journalier:
    print(f"   -> {os.path.basename(fichier)}")
    df_temp = lire_fichier_journalier(fichier)
    df_journalier = pd.concat([df_journalier, df_temp], ignore_index=True)

# ------------------------------------------------------------
# 6. NETTOYAGE ET SAUVEGARDE
# ------------------------------------------------------------
os.chdir(dossier_principal)

# --- Base annuelle ---
if not df_annuel.empty:
    df_annuel = df_annuel.drop_duplicates(subset=['Année', 'Station'])
    df_annuel = df_annuel.sort_values(['Station', 'Année'])
    df_annuel.to_csv("pluviometrie_annuelle.csv", index=False, encoding='utf-8-sig')
    print(f"\n💾 Base annuelle sauvegardée : {len(df_annuel)} lignes")

# --- Base mensuelle (issue des fichiers mensuels) ---
if not df_mensuel.empty:
    df_mensuel = df_mensuel.drop_duplicates(subset=['Année', 'Mois', 'Station'])
    df_mensuel = df_mensuel.sort_values(['Station', 'Année', 'Mois'])
    df_mensuel.to_csv("pluviometrie_mensuelle_source.csv", index=False, encoding='utf-8-sig')
    print(f"💾 Base mensuelle (source) sauvegardée : {len(df_mensuel)} lignes")

# --- Base journalière ---
if not df_journalier.empty:
    df_journalier = df_journalier.drop_duplicates(subset=['Date', 'Station'])
    df_journalier = df_journalier.sort_values(['Station', 'Date'])
    df_journalier.to_csv("pluviometrie_journaliere.csv", index=False, encoding='utf-8-sig')
    print(f"💾 Base journalière sauvegardée : {len(df_journalier)} lignes")

# ------------------------------------------------------------
# 7. AGRÉGATION MENSUELLE À PARTIR DU JOURNALIER
# ------------------------------------------------------------
if not df_journalier.empty:
    df_journalier['Année'] = df_journalier['Date'].dt.year
    df_journalier['Mois'] = df_journalier['Date'].dt.month
    df_mensuel_agrege = df_journalier.groupby(['Année', 'Mois', 'Station'])['Pluie_mm'].sum().reset_index()
    df_mensuel_agrege.to_csv("pluviometrie_mensuelle_agregee.csv", index=False, encoding='utf-8-sig')
    print(f"💾 Agrégation mensuelle (depuis journalier) sauvegardée : {len(df_mensuel_agrege)} lignes")

# ------------------------------------------------------------
# 8. FUSION : TABLE MENSUELLE COMPLÈTE
# ------------------------------------------------------------
if not df_mensuel.empty or not df_journalier.empty:
    if not df_mensuel.empty:
        source1 = df_mensuel[['Année', 'Mois', 'Station', 'Pluie_Mensuelle_mm']].copy()
        source1 = source1.rename(columns={'Pluie_Mensuelle_mm': 'Pluie_mm'})
    else:
        source1 = pd.DataFrame(columns=['Année', 'Mois', 'Station', 'Pluie_mm'])
    
    if not df_journalier.empty:
        source2 = df_mensuel_agrege.copy()
    else:
        source2 = pd.DataFrame(columns=['Année', 'Mois', 'Station', 'Pluie_mm'])
    
    df_mensuel_complet = pd.concat([source2, source1], ignore_index=True)
    df_mensuel_complet = df_mensuel_complet.drop_duplicates(subset=['Année', 'Mois', 'Station'], keep='first')
    df_mensuel_complet = df_mensuel_complet.sort_values(['Station', 'Année', 'Mois'])
    df_mensuel_complet.to_csv("pluviometrie_mensuelle_complete.csv", index=False, encoding='utf-8-sig')
    print(f"💾 Table mensuelle complète sauvegardée : {len(df_mensuel_complet)} lignes")

print("\n✅ Traitement terminé !")
print(f"Les fichiers CSV sont sauvegardés dans : {dossier_principal}")