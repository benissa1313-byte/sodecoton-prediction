import pandas as pd
import numpy as np

# Paramètres
regions = ["Garoua", "Guider", "Kaélé", "Maroua", "Mayo Galké", "Ngong", "Tchatibali", "Touboro"]
years = list(range(2000, 2027))

# Moyennes par région (valeurs réalistes)
region_params = {
    "Garoua":      {"pluie_moy": 850, "credit_moy": 140000, "surf_moy": 33000, "temp_moy": 28.5, "ph_moy": 6.6, "mat_org": 1.6, "cap_ret": 110},
    "Guider":      {"pluie_moy": 820, "credit_moy": 138000, "surf_moy": 31000, "temp_moy": 28.2, "ph_moy": 6.5, "mat_org": 1.5, "cap_ret": 105},
    "Kaélé":       {"pluie_moy": 780, "credit_moy": 135000, "surf_moy": 19000, "temp_moy": 28.8, "ph_moy": 6.7, "mat_org": 1.4, "cap_ret": 95},
    "Maroua":      {"pluie_moy": 700, "credit_moy": 142000, "surf_moy": 45000, "temp_moy": 29.5, "ph_moy": 6.4, "mat_org": 1.3, "cap_ret": 90},
    "Mayo Galké":  {"pluie_moy": 1100,"credit_moy": 145000, "surf_moy": 18000, "temp_moy": 27.5, "ph_moy": 6.8, "mat_org": 1.7, "cap_ret": 120},
    "Ngong":       {"pluie_moy": 900, "credit_moy": 136000, "surf_moy": 26000, "temp_moy": 28.0, "ph_moy": 6.5, "mat_org": 1.5, "cap_ret": 100},
    "Tchatibali":  {"pluie_moy": 750, "credit_moy": 133000, "surf_moy": 15000, "temp_moy": 29.0, "ph_moy": 6.3, "mat_org": 1.2, "cap_ret": 85},
    "Touboro":     {"pluie_moy": 1200,"credit_moy": 148000, "surf_moy": 17000, "temp_moy": 27.0, "ph_moy": 6.9, "mat_org": 1.8, "cap_ret": 115}
}

# Coefficients pour une relation linéaire forte (garantit un bon R²)
coef_pluie = 0.4
coef_credit = 0.0005
coef_temp = -8.0
coef_ph = 30.0
coef_mat_org = 40.0
coef_cap_ret = 0.6
coef_semis = -1.2
coef_densite = 0.012
coef_N = 1.8
coef_P = 1.2
coef_K = 0.9
coef_trait = 15.0
coef_superficie = 0.0005
intercept = 800

data = []
for annee in years:
    for reg in regions:
        p = region_params[reg]
        # Tendance annuelle et bruit
        pluie = p["pluie_moy"] + (annee-2000)*(-0.5) + np.random.normal(0, 40)
        pluie = max(400, min(1400, int(pluie)))
        credit = p["credit_moy"] + (annee-2000)*1200 + np.random.normal(0, 8000)
        credit = max(120000, min(200000, int(credit)))
        superficie = p["surf_moy"] + np.random.normal(0, 2000)
        superficie = max(5000, min(120000, int(superficie)))
        temp = p["temp_moy"] + np.random.normal(0, 0.8)
        temp = round(max(24, min(34, temp)), 1)
        ph = p["ph_moy"] + np.random.normal(0, 0.2)
        ph = round(max(5.0, min(8.0, ph)), 1)
        mat_org = p["mat_org"] + np.random.normal(0, 0.2)
        mat_org = round(max(0.8, min(2.5, mat_org)), 2)
        cap_ret = p["cap_ret"] + np.random.normal(0, 15)
        cap_ret = round(max(60, min(180, cap_ret)), 1)
        date_semis = 140 + np.random.normal(0, 10)
        date_semis = max(120, min(180, int(date_semis)))
        densite = 65000 + np.random.normal(0, 4000)
        densite = max(50000, min(85000, int(densite)))
        fertil_N = 60 + np.random.normal(0, 10)
        fertil_N = max(30, min(120, int(fertil_N)))
        fertil_P = 40 + np.random.normal(0, 8)
        fertil_P = max(20, min(80, int(fertil_P)))
        fertil_K = 50 + np.random.normal(0, 10)
        fertil_K = max(20, min(100, int(fertil_K)))
        nb_trait = 3 + np.random.normal(0, 1)
        nb_trait = max(1, min(8, int(nb_trait)))
        lat = 9.5 + np.random.normal(0, 0.2)
        lon = 14.0 + np.random.normal(0, 0.2)
        texture = np.random.choice(["argileux", "limono-argileux", "sablo-argileux"])
        variete = np.random.choice(["variete_A", "variete_B"])

        # Calcul linéaire du rendement
        rendement = (intercept +
                     coef_pluie * (pluie - 800) +
                     coef_credit * (credit - 130000) +
                     coef_temp * (temp - 28) +
                     coef_ph * (ph - 6.5) +
                     coef_mat_org * (mat_org - 1.5) +
                     coef_cap_ret * (cap_ret - 100) +
                     coef_semis * (date_semis - 140) +
                     coef_densite * ((densite - 65000)/1000) +
                     coef_N * (fertil_N - 60) +
                     coef_P * (fertil_P - 40) +
                     coef_K * (fertil_K - 50) +
                     coef_trait * (nb_trait - 3) +
                     coef_superficie * ((superficie - 30000)/1000))
        rendement += np.random.normal(0, 35)
        rendement = max(500, min(2500, int(rendement)))

        data.append([reg, annee, pluie, credit, rendement, superficie,
                     temp, ph, mat_org, cap_ret, texture,
                     date_semis, densite, fertil_N, fertil_P, fertil_K, nb_trait, variete, lat, lon])

# Création du DataFrame
columns = [
    "Région", "Année", "Pluie_Annuelle_mm", "Credit_intrant_ha_FCFA", "Rdt_region_kg_ha", "Superficie_ha",
    "Temp_Moyenne_C", "pH", "Matiere_Organique_pourcent", "Capacite_Retention_mm", "Texture",
    "Date_Semis_jour", "Densite_plants_ha", "Fertil_N_kg_ha", "Fertil_P_kg_ha", "Fertil_K_kg_ha",
    "Nb_Traitements", "Variete", "Latitude", "Longitude"
]

df = pd.DataFrame(data, columns=columns)

# Sauvegarde en CSV
df.to_csv("dataset_complet_2000_2026.csv", index=False, encoding='utf-8-sig')
# Sauvegarde en Excel
df.to_excel("dataset_complet_2000_2026.xlsx", index=False, engine='openpyxl')

print(f"✅ Dataset généré : {len(df)} lignes, années {df['Année'].min()} à {df['Année'].max()}")
print("   Fichiers créés : dataset_complet_2000_2026.csv  et  dataset_complet_2000_2026.xlsx")