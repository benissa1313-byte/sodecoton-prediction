# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:29:07 2026

@author: THIS PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Lecture du dataset (à adapter selon votre format)
# Utilisez l'un des deux :
df = pd.read_csv("dataset_complet_2000_2026.csv", encoding='utf-8-sig')
# ou
# df = pd.read_excel("dataset_complet_2000_2026.xlsx", sheet_name="Sheet1")

# Colonnes à ignorer (identifiants, cible)
identifiants = ["Région", "Année"]
target = "Rdt_region_kg_ha"

# Création du dossier principal
base_dir = Path("analyse_variables")
base_dir.mkdir(exist_ok=True)

# Liste des variables (toutes les colonnes sauf identifiants et cible)
variables = [col for col in df.columns if col not in identifiants + [target]]

for var in variables:
    print(f"Traitement de : {var}")
    var_dir = base_dir / var
    var_dir.mkdir(exist_ok=True)

    # --- 1. Export des données (Région, Année, variable) ---
    data_export = df[["Région", "Année", var]].copy()
    csv_path = var_dir / f"{var}_data.csv"
    excel_path = var_dir / f"{var}_data.xlsx"
    data_export.to_csv(csv_path, index=False, encoding='utf-8-sig')
    data_export.to_excel(excel_path, index=False)

    # --- 2. Graphiques adaptés au type de variable ---
    if pd.api.types.is_numeric_dtype(df[var]):
        # Variable numérique
        # 2a. Histogramme + KDE
        fig, ax = plt.subplots()
        sns.histplot(df[var].dropna(), bins=30, kde=True, color='#2e7d32', ax=ax)
        ax.set_title(f"Distribution de {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Fréquence")
        plt.tight_layout()
        plt.savefig(var_dir / f"histogramme_{var}.png", dpi=150)
        plt.close()

        # 2b. Boxplot par région
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="Région", y=var, palette='Greens', ax=ax)
        ax.set_title(f"Boxplot de {var} par région")
        ax.set_xlabel("Région")
        ax.set_ylabel(var)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(var_dir / f"boxplot_region_{var}.png", dpi=150)
        plt.close()

        # 2c. Évolution temporelle (moyenne par année)
        if "Année" in df.columns:
            evolution = df.groupby("Année")[var].mean().reset_index()
            fig, ax = plt.subplots()
            ax.plot(evolution["Année"], evolution[var], marker='o', color='#1b5e20')
            ax.set_title(f"Évolution de {var} (moyenne nationale)")
            ax.set_xlabel("Année")
            ax.set_ylabel(var)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(var_dir / f"evolution_temporelle_{var}.png", dpi=150)
            plt.close()

        # 2d. Corrélation avec la cible (nuage de points + droite de régression)
        if var != target:
            fig, ax = plt.subplots()
            ax.scatter(df[var], df[target], alpha=0.5, color='#2e7d32')
            # Droite de régression
            from scipy import stats
            mask = df[[var, target]].dropna()
            if len(mask) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(mask[var], mask[target])
                x_vals = np.linspace(mask[var].min(), mask[var].max(), 100)
                y_vals = slope * x_vals + intercept
                ax.plot(x_vals, y_vals, 'r--', label=f"R² = {r_value**2:.3f}")
                ax.legend()
            ax.set_xlabel(var)
            ax.set_ylabel(target)
            ax.set_title(f"Corrélation entre {var} et {target}")
            plt.tight_layout()
            plt.savefig(var_dir / f"correlation_{var}_vs_{target}.png", dpi=150)
            plt.close()

    else:
        # Variable catégorielle (texte)
        # 2a. Diagramme en barres des fréquences
        fig, ax = plt.subplots()
        df[var].value_counts().plot(kind='bar', color='#2e7d32', ax=ax)
        ax.set_title(f"Fréquences des modalités de {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Effectif")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(var_dir / f"barplot_{var}.png", dpi=150)
        plt.close()

        # 2b. Répartition par région (empilement ou heatmap)
        cross = pd.crosstab(df["Région"], df[var], normalize='index')
        fig, ax = plt.subplots(figsize=(12, 6))
        cross.plot(kind='bar', stacked=True, ax=ax, colormap='Greens')
        ax.set_title(f"Répartition de {var} par région")
        ax.set_xlabel("Région")
        ax.set_ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(var_dir / f"stacked_region_{var}.png", dpi=150)
        plt.close()

        # 2c. Évolution temporelle (proportions annuelles)
        if "Année" in df.columns:
            cross_year = pd.crosstab(df["Année"], df[var], normalize='index')
            fig, ax = plt.subplots(figsize=(14, 6))
            cross_year.plot(kind='area', stacked=True, ax=ax, colormap='Greens', alpha=0.7)
            ax.set_title(f"Évolution de {var} (proportions annuelles)")
            ax.set_xlabel("Année")
            ax.set_ylabel("Proportion")
            plt.legend(title=var, bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            plt.savefig(var_dir / f"evolution_temporelle_{var}.png", dpi=150)
            plt.close()

    print(f"  -> Dossier {var_dir} créé avec {len(list(var_dir.glob('*')))} fichiers.")

print("\nAnalyse terminée. Tous les dossiers et fichiers ont été générés dans 'analyse_variables'.")