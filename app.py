import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from moteur_recherche import (
    clean_text, 
    recherche_textuelle, 
    charger_fichier, 
    charger_stopwords
)

# Configuration de la page
st.set_page_config(
    page_title="Moteur de Recherche de Pièces",
    page_icon="🔍",
    layout="wide"
)

# Style personnalisé
st.markdown("""
<style>
    .main {
        padding: 1rem 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .block-container {
        max-width: 1200px;
    }
    .result-table {
        font-size: 0.9rem;
    }
    .highlight {
        background-color: #e6f3ff;
        font-weight: bold;
    }
    .score-bar {
        height: 10px;
        background-color: #0068c9;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🔍 Moteur de Recherche de Pièces")
st.markdown("Recherchez des pièces par description et obtenez des résultats triés par pertinence.")

# Fonction pour créer un lien de téléchargement
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {text}</a>'
    return href

# Sidebar pour le chargement de fichiers et les paramètres
with st.sidebar:
    st.header("Configuration")
    
    # Chargement du fichier
    uploaded_file = st.file_uploader("Charger un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Fichier chargé avec succès! ({len(df)} lignes)")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
            df = None
    else:
        # Vérifier si un fichier par défaut existe
        default_file = "data/data_filtre_v2.csv"
        if os.path.exists(default_file):
            df = charger_fichier(default_file)
            st.info(f"Fichier par défaut chargé: {default_file} ({len(df)} lignes)")
        else:
            st.warning("Aucun fichier chargé. Veuillez charger un fichier CSV.")
            df = None
    
    # Paramètres de recherche
    st.subheader("Paramètres de recherche")
    
    # Colonnes à rechercher
    if df is not None:
        colonnes_disponibles = [col for col in df.columns if "Désignation" in col]
        colonnes_recherche = st.multiselect(
            "Colonnes à rechercher",
            options=colonnes_disponibles,
            default=colonnes_disponibles
        )
    else:
        colonnes_recherche = []
    
    # Nombre de résultats à afficher
    nb_resultats = st.slider("Nombre de résultats à afficher", 5, 100, 10)
    
    # Seuil de score minimum
    seuil_score = st.slider("Seuil de score minimum", 0, 100, 10)
    
    # Poids des colonnes
    st.subheader("Poids des colonnes")
    if df is not None and len(colonnes_recherche) > 0:
        poids_colonnes = {}
        for col in colonnes_recherche:
            default_weight = 1.0 if col == "Désignation 1" else (0.4 if col == "Désignation 2" else 0.2)
            poids_colonnes[col] = st.slider(f"Poids de {col}", 0.0, 1.0, default_weight, 0.1)

# Barre de recherche
requete = st.text_input("Rechercher une pièce", placeholder="Ex: clé à oeil cliquet avec entrainement 17mm")

# Affichage des statistiques du jeu de données
if df is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de pièces", f"{len(df):,}")
    
    # Compter les familles
    if "Famille" in df.columns:
        familles = df["Famille"].dropna().nunique()
        col2.metric("Nombre de familles", f"{familles:,}")
    
    # Compter les sections
    if "Section" in df.columns:
        sections = df["Section"].dropna().nunique()
        col3.metric("Nombre de sections", f"{sections:,}")

# Recherche
if requete and df is not None and len(colonnes_recherche) > 0:
    with st.spinner("Recherche en cours..."):
        # Nettoyage de la requête
        try:
            stopwords_list = charger_stopwords()
        except:
            stopwords_list = []
        
        requete_nettoyee = clean_text(requete, stopwords_list)
        st.caption(f"Requête nettoyée: **{requete_nettoyee}**")
        
        # Recherche
        resultats = recherche_textuelle(df, requete_nettoyee, colonnes=colonnes_recherche)
        
        # Recalcul du score global avec les poids personnalisés
        if 'poids_colonnes' in locals():
            resultats['score_global'] = 0
            for col in colonnes_recherche:
                resultats['score_global'] += resultats[f'score_{col}'] * poids_colonnes[col]
            
            # Retrier les résultats avec les nouveaux scores
            resultats = resultats.sort_values(by='score_global', ascending=False)
        
        # Filtrage par seuil de score
        resultats = resultats[resultats['score_global'] >= seuil_score]
        
        # Limiter le nombre de résultats
        resultats = resultats.head(nb_resultats)
        
        # Affichage des résultats
        st.subheader(f"Résultats de recherche ({len(resultats)} trouvés)")
        
        if len(resultats) > 0:
            # Colonnes à afficher
            colonnes_affichage = ["Code Pièce"] + colonnes_recherche
            
            if "Famille" in resultats.columns:
                colonnes_affichage.append("Famille")
            
            if "Prix" in resultats.columns:
                colonnes_affichage.append("Prix")
            
            # Créer une colonne pour la barre de score
            resultats['Score'] = resultats['score_global'].apply(
                lambda x: f'<div class="score-bar" style="width: {min(100, int(x))}%;"></div> {x:.1f}'
            )
            
            # Ajouter la colonne de score à l'affichage
            colonnes_affichage.append("Score")
            
            # Afficher le tableau des résultats
            st.markdown(
                resultats[colonnes_affichage].to_html(
                    escape=False,
                    formatters={'Score': lambda x: x},
                    classes='result-table',
                    index=False
                ),
                unsafe_allow_html=True
            )
            
            # Bouton de téléchargement des résultats
            st.markdown(
                get_download_link(resultats, "resultats_recherche.csv", "Télécharger les résultats"),
                unsafe_allow_html=True
            )
            
            # Afficher les détails du premier résultat
            with st.expander("Voir les détails du meilleur résultat"):
                meilleur = resultats.iloc[0]
                st.markdown(f"### {meilleur['Code Pièce']}")
                
                cols = st.columns(3)
                for i, col_name in enumerate(colonnes_recherche):
                    j = i % 3
                    with cols[j]:
                        st.markdown(f"**{col_name}**")
                        st.markdown(f"{meilleur[col_name]}")
                
                st.markdown("### Scores de pertinence")
                
                score_cols = st.columns(len(colonnes_recherche) + 1)
                for i, col_name in enumerate(colonnes_recherche):
                    with score_cols[i]:
                        score_value = meilleur[f'score_{col_name}']
                        st.metric(col_name, f"{score_value:.1f}")
                
                with score_cols[-1]:
                    st.metric("Score Global", f"{meilleur['score_global']:.1f}", delta="Total")
        else:
            st.info("Aucun résultat ne correspond à votre recherche. Essayez d'autres termes ou réduisez le seuil de score.")
else:
    if df is not None:
        st.info("Entrez votre recherche dans le champ ci-dessus.")
    
# Affichage des informations sur le fonctionnement
with st.expander("Comment fonctionne ce moteur de recherche ?"):
    st.markdown("""
    ## Fonctionnement du moteur de recherche
    
    Ce moteur de recherche utilise plusieurs techniques pour trouver les résultats les plus pertinents :
    
    1. **Nettoyage des données** : Les termes de recherche sont normalisés (majuscules, sans accents) et les mots vides ("le", "la", "de", etc.) sont supprimés.
    
    2. **Recherche intelligente** :
       - Gestion des variations linguistiques (singulier/pluriel)
       - Tolérance aux fautes d'orthographe
       - Analyse des similarités entre mots
       
    3. **Système de score** :
       - Score pour chaque colonne de désignation
       - Pondération des colonnes (Désignation 1 compte plus que Désignation 2, etc.)
       - Bonus pour les mots trouvés en début de texte
       - Bonus pour l'ordre des mots
       
    4. **Filtrage et tri** :
       - Les résultats sont filtrés selon un seuil de score minimum
       - Tri par pertinence (score global)
    """)

# Pied de page
st.markdown("---")
st.markdown("Moteur de recherche de pièces développé avec Streamlit et Python.")