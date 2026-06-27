# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:48:50 2026

@author: THIS PC
"""

import pandas as pd
import numpy as np
import glob
import os
import re

# ------------------------------------------------------------
# 1. Détection du dossier
# ------------------------------------------------------------
bureau = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(bureau):
    bureau = os.path.join(os.path.expanduser("~"), "Bureau")
dossier = os.path.join(bureau, "pluviometrie", "mensuel")
if not os.path.exists(dossier):
    dossier = input("Chemin complet du dossier 'mensuel' : ").strip()
print(f"📁 Dossier : {dossier}")
os.chdir(dossier)

# ------------------------------------------------------------
# 2. Correspondance des stations (noms exacts en majuscules)
# ------------------------------------------------------------
CORRESP = {
    'MAROUA': 'Maroua',
    'KAELE': 'Kaélé',
    'TCHATIBALI': 'Tchatibali',
    'TBLI': 'Tchatibali',
    'GUIDER': 'Guider',
    'GAROUA': 'Garoua',
    'NGONG': 'Ngong',
    'POLI': 'Poli',
    'M-GALKE': 'Mayo Galké',
    'MAYO-GALKE': 'Mayo Galké',
    'TOUBORO': 'Touboro',
    'TOUBOURO': 'Touboro',
    'HOME': 'Homé'
}

# ------------------------------------------------------------
# 3. Extraction du mois depuis le nom de la feuille
#    et de l'année depuis la ligne "MOIS DE ..."
# ------------------------------------------------------------
def extraire_mois(nom_feuille):
    mois_map = {
        'JANVIER':1, 'FEVRIER':2, 'MARS':3, 'AVRIL':4, 'MAI':5, 'JUIN':6,
        'JUILLET':7, 'AOUT':8, 'AOÛT':8, 'SEPTEMBRE':9, 'OCTOBRE':10,
        'NOVEMBRE':11, 'DECEMBRE':12
    }
    nom = nom_feuille.upper().strip()
    for mot, num in mois_map.items():
        if mot == nom or mot in nom:
            return num
    return None

def extraire_annee_depuis_texte(texte):
    """Cherche une année 20xx dans une chaîne."""
    match = re.search(r'20\d{2}', str(texte))
    if match:
        return int(match.group())
    return None

# ------------------------------------------------------------
# 4. Lecture d'un fichier mensuel
# ------------------------------------------------------------
def lire_fichier(chemin):
    xl = pd.ExcelFile(chemin)
    lignes = []
    for sheet in xl.sheet_names:
        if sheet.upper() in ['IMPRIME', 'FEUILLE12', 'FEUILLE13']:
            continue
        mois = extraire_mois(sheet)
        if mois is None:
            continue
        
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        
        # Chercher l'année dans la ligne qui contient "MOIS DE ..."
        annee = None
        for i in range(min(10, len(df))):
            texte = str(df.iloc[i, 0])
            if 'MOIS DE' in texte.upper():
                annee = extraire_annee_depuis_texte(texte)
                break
        # Si pas trouvé, essayer de prendre l'année du nom de fichier
        if annee is None:
            match = re.search(r'20\d{2}', os.path.basename(chemin))
            if match:
                annee = int(match.group())
        if annee is None:
            print(f"      ⚠️ Année non trouvée pour la feuille {sheet}")
            continue
        
        # Ligne "STATION" (recherche tolérante)
        ligne_station = None
        for i in range(min(20, len(df))):
            if 'STATION' in str(df.iloc[i, 0]).upper():
                ligne_station = i
                break
        if ligne_station is None:
            continue
        
        # Jours (colonnes 1 à 31)
        jours_vals = []
        for c in range(1, 32):
            val = df.iloc[ligne_station, c]
            try:
                j = int(float(val))
                if 1 <= j <= 31:
                    jours_vals.append(j)
                else:
                    jours_vals.append(np.nan)
            except:
                jours_vals.append(np.nan)
        
        # Lire les stations
        nb_extrait = 0
        for row_idx in range(ligne_station+1, min(ligne_station+15, len(df))):
            nom_brut = str(df.iloc[row_idx, 0]).strip().upper()
            if nom_brut not in CORRESP:
                continue
            station = CORRESP[nom_brut]
            for col_idx, jour in enumerate(jours_vals, start=1):
                if pd.isna(jour):
                    continue
                val = df.iloc[row_idx, col_idx]
                if pd.isna(val) or str(val).strip() == '':
                    continue
                try:
                    pluie = float(val)
                    if pluie == 0:
                        continue  # optionnel : ignorer les zéros
                except:
                    continue
                date = pd.Timestamp(year=annee, month=mois, day=int(jour))
                lignes.append({'Date': date, 'Station': station, 'Pluie_mm': pluie})
                nb_extrait += 1
        if nb_extrait > 0:
            print(f"      Feuille {sheet} : {nb_extrait} valeurs")
    return pd.DataFrame(lignes)

# ------------------------------------------------------------
# 5. Parcours de tous les fichiers Excel du dossier
# ------------------------------------------------------------
fichiers = glob.glob("*.xlsx") + glob.glob("*.xls")
print(f"Fichiers trouvés : {fichiers}")

df_total = pd.DataFrame()
for f in fichiers:
    print(f"   Traitement de {os.path.basename(f)}...")
    temp = lire_fichier(f)
    if not temp.empty:
        df_total = pd.concat([df_total, temp], ignore_index=True)

# ------------------------------------------------------------
# 6. Sauvegarde
# ------------------------------------------------------------
if not df_total.empty:
    df_total = df_total.drop_duplicates(['Date','Station'])
    df_total = df_total.sort_values(['Station','Date'])
    df_total.to_csv("pluviometrie_journaliere.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ Journalier : {len(df_total)} lignes")

    # Agrégation mensuelle
    df_total['Année'] = df_total['Date'].dt.year
    df_total['Mois'] = df_total['Date'].dt.month
    df_mensuel = df_total.groupby(['Année','Mois','Station'])['Pluie_mm'].sum().reset_index()
    df_mensuel.to_csv("pluviometrie_mensuelle_agregee.csv", index=False, encoding='utf-8-sig')
    print(f"✅ Mensuel agrégé : {len(df_mensuel)} lignes")
else:
    print("\n❌ Aucune donnée extraite.")