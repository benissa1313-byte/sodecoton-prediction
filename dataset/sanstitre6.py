# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:50:55 2026

@author: THIS PC
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------
bureau = Path.home() / "Desktop"
file_path = bureau / "Projet de stage" / "dataset" / "dataset_complet.xlsx"
df = pd.read_excel(file_path, sheet_name="Consolide")

print("=== Dataset consolidé ===")
print(f"Dimensions : {df.shape}")
print(f"Colonnes : {df.columns.tolist()}")
print(df.head(3))

# ------------------------------------------------------------
# 2. Nettoyage léger
# ------------------------------------------------------------
# Convertir les colonnes booléennes (True/False) en 0/1 pour les corrélations
bool_cols = df.select_dtypes(include='bool').columns
for col in bool_cols:
    df[col] = df[col].astype(int)

# Vérifier les types
print("\nTypes après conversion :")
print(df.dtypes)

# ------------------------------------------------------------
# 3. Statistiques descriptives
# ------------------------------------------------------------
print("\n=== Statistiques descriptives (variables clés) ===")
key_vars = ["Rdt_region_kg_ha", "Production_region_T", "S_realise", 
            "Pluie_Annuelle_mm", "NbTotalGrosProducteur", "NbTotalGrpt",
            "prix_moyen_article", "Tonnage_S_realise", "Tonnage_Credit"]
available_vars = [v for v in key_vars if v in df.columns]
print(df[available_vars].describe())

# ------------------------------------------------------------
# 4. Analyse par région et année
# ------------------------------------------------------------
# Rendement moyen par région (toutes années confondues)
print("\n=== Rendement moyen par région ===")
region_rdt = df.groupby("Région")["Rdt_region_kg_ha"].mean().sort_values(ascending=False)
print(region_rdt)

# Production totale par région
print("\n=== Production totale par région (Tonnes) ===")
region_prod = df.groupby("Région")["Production_region_T"].sum().sort_values(ascending=False)
print(region_prod)

# Évolution du rendement pour les principales régions
top_regions = region_rdt.head(5).index.tolist()
df_top = df[df["Région"].isin(top_regions)]

# ------------------------------------------------------------
# 5. Visualisations
# ------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 5.1 Évolution du rendement par région (top 5)
plt.figure()
for region in top_regions:
    subset = df[df["Région"] == region].sort_values("Année")
    plt.plot(subset["Année"], subset["Rdt_region_kg_ha"], marker='o', label=region)
plt.title("Évolution du rendement (kg/ha) – Top 5 régions")
plt.xlabel("Année")
plt.ylabel("Rendement (kg/ha)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(bureau / "Projet de stage" / "dataset" / "evolution_rendement_regions.png", dpi=150)
plt.show()

# 5.2 Boxplot des rendements par région
plt.figure()
sns.boxplot(data=df, x="Région", y="Rdt_region_kg_ha", palette="Set2")
plt.title("Distribution du rendement par région")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(bureau / "Projet de stage" / "dataset" / "boxplot_rendement_region.png", dpi=150)
plt.show()

# 5.3 Corrélation entre variables numériques (version propre et lisible)
# Sélection des variables pertinentes (celles qui ont un sens agronomique ou économique)
variables_pertinentes = [
    "Rdt_region_kg_ha", "Production_region_T", "S_realise",
    "Pluie_Annuelle_mm", "Credit_intrant_ha_FCFA",
    "NbTotalGrpt", "prix_moyen_article"
]
# Ajouter NbTotalGrosProducteur s'il existe
if "NbTotalGrosProducteur" in df.columns:
    variables_pertinentes.append("NbTotalGrosProducteur")

# Ne garder que les colonnes réellement présentes
variables_pertinentes = [v for v in variables_pertinentes if v in df.columns]

if len(variables_pertinentes) >= 2:
    corr_matrix = df[variables_pertinentes].corr()
    
    plt.figure(figsize=(10, 8))
    # Masquer la partie supérieure (triangulaire) pour éviter les doublons
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, mask=mask,
                annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
    plt.title("Matrice de corrélation (variables clés)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(bureau / "Projet de stage" / "dataset" / "correlation_matrix_propre.png", dpi=150)
    plt.show()
else:
    print("Pas assez de variables pertinentes pour la matrice de corrélation.")

# 5.4 Relation pluviométrie vs rendement
plt.figure()
sns.scatterplot(data=df, x="Pluie_Annuelle_mm", y="Rdt_region_kg_ha", hue="Région", size="Production_region_T", sizes=(20, 200), alpha=0.7)
plt.title("Pluie annuelle vs Rendement")
plt.xlabel("Pluie (mm)")
plt.ylabel("Rendement (kg/ha)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(bureau / "Projet de stage" / "dataset" / "pluie_vs_rendement.png", dpi=150)
plt.show()

# 5.5 Impact des intrants (Credit_intrant_ha_FCFA vs rendement)
plt.figure()
sns.regplot(data=df, x="Credit_intrant_ha_FCFA", y="Rdt_region_kg_ha", scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Crédit intrant par hectare vs Rendement")
plt.xlabel("Crédit intrant (FCFA/ha)")
plt.ylabel("Rendement (kg/ha)")
plt.tight_layout()
plt.savefig(bureau / "Projet de stage" / "dataset" / "credit_vs_rendement.png", dpi=150)
plt.show()

# 5.6 Évolution de la production totale par année (toutes régions)
prod_par_an = df.groupby("Année")["Production_region_T"].sum()
plt.figure()
prod_par_an.plot(kind="bar", color="teal")
plt.title("Production totale de coton par année (Toutes régions)")
plt.xlabel("Année")
plt.ylabel("Production (Tonnes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(bureau / "Projet de stage" / "dataset" / "production_totale_annee.png", dpi=150)
plt.show()

# 5.7 Distribution du rendement
plt.figure()
sns.histplot(df["Rdt_region_kg_ha"], bins=20, kde=True, color="purple")
plt.title("Distribution du rendement (kg/ha)")
plt.xlabel("Rendement")
plt.tight_layout()
plt.savefig(bureau / "Projet de stage" / "dataset" / "distribution_rendement.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 6. Export du dataset final (CSV)
# ------------------------------------------------------------
output_csv = bureau / "Projet de stage" / "dataset" / "dataset_analyse.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ Dataset final exporté en CSV : {output_csv}")

# Optionnel : exporter un sous-ensemble pour modélisation
output_model = bureau / "Projet de stage" / "dataset" / "dataset_modelisation.csv"
df_model = df[["Année", "Région", "Rdt_region_kg_ha", "Production_region_T", "S_realise",
               "Pluie_Annuelle_mm", "Credit_intrant_ha_FCFA", "NbTotalGrpt", "prix_moyen_article"]]
df_model.to_csv(output_model, index=False, encoding='utf-8-sig')
print(f"✅ Dataset pour modélisation : {output_model}")