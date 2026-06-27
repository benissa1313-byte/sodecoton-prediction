# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:48:30 2026

@author: THIS PC
"""

# =============================================================================
# SCRIPT 1 : TRAITEMENT DES FICHIERS ANNUELS
# À exécuter dans Spyder
# =============================================================================
import pandas as pd
import glob
import os

# --- 1. Détection automatique du dossier "annuel" ---
bureau = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(bureau):
    bureau = os.path.join(os.path.expanduser("~"), "Bureau")

dossier_annuel = os.path.join(bureau, "pluviometrie", "annuel")

if not os.path.exists(dossier_annuel):
    print(f"❌ Dossier introuvable : {dossier_annuel}")
    chemin = input("Entrez le chemin complet du dossier 'annuel' : ").strip()
    if os.path.exists(chemin):
        dossier_annuel = chemin
    else:
        raise FileNotFoundError("Dossier non trouvé.")

print(f"📁 Dossier de travail : {dossier_annuel}")
os.chdir(dossier_annuel)

# --- 2. Lecture des fichiers Excel du dossier ---
fichiers = glob.glob("*.xlsx") + glob.glob("*.xls")
print(f"Fichiers trouvés : {fichiers}")

def lire_annuel(chemin):
    df = pd.read_excel(chemin, sheet_name=0)  # première feuille
    # Renommer la première colonne en "Année"
    df = df.rename(columns={df.columns[0]: 'Année'})
    # Garder les lignes avec une année valide
    df = df[df['Année'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]
    df['Année'] = df['Année'].astype(int)
    # Transformation en format long
    df_long = df.melt(id_vars='Année', var_name='Station', value_name='Pluie_Annuelle_mm')
    df_long['Station'] = df_long['Station'].str.strip()
    df_long = df_long.dropna(subset=['Pluie_Annuelle_mm'])
    return df_long

df_annuel = pd.DataFrame()
for f in fichiers:
    print(f"   Traitement de {f}...")
    df_temp = lire_annuel(f)
    df_annuel = pd.concat([df_annuel, df_temp], ignore_index=True)

# --- 3. Nettoyage et sauvegarde ---
if not df_annuel.empty:
    df_annuel = df_annuel.drop_duplicates(subset=['Année', 'Station'])
    df_annuel = df_annuel.sort_values(['Station', 'Année'])
    df_annuel.to_csv("pluviometrie_annuelle.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Fichier sauvegardé : pluviometrie_annuelle.csv ({len(df_annuel)} lignes)")
else:
    print("⚠️ Aucune donnée annuelle trouvée.")