# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:49:24 2026

@author: THIS PC
"""
import pandas as pd
import glob
import os
import re

# ------------------------------------------------------------
# 1. Détection du dossier "journalier"
# ------------------------------------------------------------
bureau = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(bureau):
    bureau = os.path.join(os.path.expanduser("~"), "Bureau")

dossier_journalier = os.path.join(bureau, "pluviometrie", "journalier")

if not os.path.exists(dossier_journalier):
    print(f"❌ Dossier introuvable : {dossier_journalier}")
    chemin = input("Entrez le chemin complet du dossier 'journalier' : ").strip()
    if os.path.exists(chemin):
        dossier_journalier = chemin
    else:
        raise FileNotFoundError("Dossier non trouvé.")

print(f"📁 Dossier de travail : {dossier_journalier}")
os.chdir(dossier_journalier)

# ------------------------------------------------------------
# 2. Correspondance des noms de stations
# ------------------------------------------------------------
CORRESP = {
    'MAROUA': 'Maroua',
    'KAELE': 'Kaélé',
    'TCHATIBALI': 'Tchatibali',
    'GUIDER': 'Guider',
    'GAROUA': 'Garoua',
    'NGONG': 'Ngong',
    'POLI': 'Poli',
    'MAYO-GALKE': 'Mayo Galké',
    'TOUBOURO': 'Touboro',
    'TOUBORO': 'Touboro',
    'HOME': 'Homé'
}

# ------------------------------------------------------------
# 3. Fonction de lecture d'un fichier quotidien
# ------------------------------------------------------------
def lire_fichier_quotidien(chemin):
    xl = pd.ExcelFile(chemin)
    lignes = []
    for sheet in xl.sheet_names:
        # Détecter les feuilles nommées JJ-MM-AA
        match = re.match(r'(\d{2})-(\d{2})-(\d{2})', sheet)
        if not match:
            continue
        jour, mois, annee_abr = match.groups()
        jour = int(jour)
        mois = int(mois)
        annee = 2000 + int(annee_abr)  # suppose que l'année est 20xx
        date = pd.Timestamp(year=annee, month=mois, day=jour)
        
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        
        # Parcourir toutes les cellules pour trouver les stations connues
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                cell_value = str(df.iloc[r, c]).strip().upper()
                if cell_value in CORRESP:
                    # Chercher la valeur de pluie dans la cellule immédiatement à droite
                    if c + 1 < df.shape[1]:
                        val = df.iloc[r, c + 1]
                        if pd.notna(val) and str(val).strip() != '':
                            try:
                                pluie = float(val)
                                lignes.append({
                                    'Date': date,
                                    'Station': CORRESP[cell_value],
                                    'Pluie_mm': pluie
                                })
                            except:
                                pass
    return pd.DataFrame(lignes)

# ------------------------------------------------------------
# 4. Parcours de tous les fichiers Excel du dossier
# ------------------------------------------------------------
fichiers = glob.glob("*.xlsx") + glob.glob("*.xls")
print(f"Fichiers trouvés : {fichiers}")

df_total = pd.DataFrame()
for f in fichiers:
    print(f"   Traitement de {f}...")
    temp = lire_fichier_quotidien(f)
    print(f"      -> {len(temp)} lignes extraites")
    df_total = pd.concat([df_total, temp], ignore_index=True)

# ------------------------------------------------------------
# 5. Nettoyage et sauvegarde
# ------------------------------------------------------------
if not df_total.empty:
    df_total = df_total.drop_duplicates(subset=['Date', 'Station'])
    df_total = df_total.sort_values(['Station', 'Date'])
    df_total.to_csv("pluviometrie_journaliere.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ Fichier sauvegardé : pluviometrie_journaliere.csv ({len(df_total)} lignes)")
    
    # Agrégation mensuelle optionnelle
    df_total['Année'] = df_total['Date'].dt.year
    df_total['Mois'] = df_total['Date'].dt.month
    df_mensuel = df_total.groupby(['Année', 'Mois', 'Station'])['Pluie_mm'].sum().reset_index()
    df_mensuel.to_csv("pluviometrie_mensuelle_agregee.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Agrégation mensuelle sauvegardée : pluviometrie_mensuelle_agregee.csv ({len(df_mensuel)} lignes)")
else:
    print("\n❌ Aucune donnée extraite. Vérifiez la structure des fichiers.")