# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:01:43 2026

@author: THIS PC
"""

import pandas as pd
import numpy as np
import glob
import os
import re

bureau = os.path.join(os.path.expanduser("~"), "Desktop")
dossier = os.path.join(bureau, "pluviometrie", "journalier")
os.chdir(dossier)
print("📁", dossier)

CORRESP = {
    'MAROUA':'Maroua','KAELE':'Kaélé','TCHATIBALI':'Tchatibali','TBLI':'Tchatibali',
    'GUIDER':'Guider','GAROUA':'Garoua','NGONG':'Ngong','MAYO-GALKE':'Mayo Galké',
    'M-GALKE':'Mayo Galké','POLI':'Poli','TOUBORO':'Touboro','TOUBOURO':'Touboro','HOME':'Homé'
}

def extraire_mois_annee(nom):
    mois_map = {'JANVIER':1,'FEVRIER':2,'MARS':3,'AVRIL':4,'MAI':5,'JUIN':6,
                'JUILLET':7,'AOUT':8,'AOÛT':8,'SEPTEMBRE':9,'OCTOBRE':10,'NOVEMBRE':11,'DECEMBRE':12}
    for mot, num in mois_map.items():
        if mot in nom.upper():
            annee_match = re.search(r'20\d{2}', nom)
            if annee_match:
                return num, int(annee_match.group())
    return None, None

def lire_fichier(chemin):
    xl = pd.ExcelFile(chemin)
    lignes = []
    for sheet in xl.sheet_names:
        if sheet.upper() in ['IMPRIME','FEUILLE12','FEUILLE13']:
            continue
        mois, annee = extraire_mois_annee(sheet)
        if mois is None:
            continue
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        # Chercher la ligne qui contient les jours (1,2,3...)
        ligne_jours = None
        for i in range(min(15, len(df))):
            row = df.iloc[i, 1:32].values
            if sum(1 for v in row if pd.notna(v) and str(v).isdigit() and 1 <= int(float(v)) <= 31) >= 10:
                ligne_jours = i
                break
        if ligne_jours is None:
            continue
        # Jours valides
        jours_vals = []
        for v in df.iloc[ligne_jours, 1:32]:
            try:
                j = int(float(v))
                if 1 <= j <= 31:
                    jours_vals.append(j)
                else:
                    jours_vals.append(np.nan)
            except:
                jours_vals.append(np.nan)
        # Lire les stations sur les lignes suivantes
        for row_idx in range(ligne_jours+1, min(ligne_jours+15, len(df))):
            nom_brut = str(df.iloc[row_idx, 0]).strip().upper()
            if nom_brut not in CORRESP:
                continue
            station = CORRESP[nom_brut]
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
                lignes.append({
                    'Date': pd.Timestamp(year=annee, month=mois, day=int(jour)),
                    'Station': station,
                    'Pluie_mm': pluie
                })
    return pd.DataFrame(lignes)

fichiers = glob.glob("*.xlsx")
df_total = pd.DataFrame()
for f in fichiers:
    print(f"Traitement de {f}...")
    temp = lire_fichier(f)
    print(f"   -> {len(temp)} lignes")
    df_total = pd.concat([df_total, temp])

if not df_total.empty:
    df_total.to_csv("pluviometrie_journaliere.csv", index=False)
    print("✅ Sauvegardé.")
else:
    print("❌ Échec.")