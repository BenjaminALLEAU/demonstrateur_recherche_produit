import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from moteur_recherche_optimise import (
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
    .highlight {
        background-color: #e6f3ff;
        font-weight: bold;
    }
    .score-bar {
        height: 10px;
        background-color: #0068c9;
        border-radius: 5px;
        margin-top: 5px;
    }
    .stDataFrame {
        font-size: 14px;
    }
    .dataframe-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-eczf16 {
        max-height: 600px;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("Moteur de Recherche de Pièces")
st.markdown("Recherchez des pièces par description et obtenez des résultats triés par pertinence.")

# Fonction pour créer un lien de téléchargement
@st.cache_data
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {text}</a>'
    return href

# Fonction pour charger les données avec mise en cache
@st.cache_resource
def load_data(file_path):
    return charger_fichier(file_path)

# Fonction pour formater le score avec une barre visuelle
def format_score(score):
    return f"<div style='width:100%'>{score:.1f}<div class='score-bar' style='width: {min(100, int(score))}%;'></div></div>"

# Sidebar pour le chargement de fichiers et les paramètres
with st.sidebar:
    st.header("Configuration")
    
    # Chargement du fichier
    uploaded_file = st.file_uploader("Charger un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Enregistrer le fichier téléchargé temporairement
            with open("temp_uploaded.csv", "wb") as f:
                f.write(uploaded_file.getvalue())
            df = load_data("temp_uploaded.csv")
            st.success(f"Fichier chargé avec succès! ({len(df)} lignes)")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
            df = None
    else:
        # Vérifier si un fichier par défaut existe
        default_file = "data/data_filtre_v2.csv"
        if os.path.exists(default_file):
            df = load_data(default_file)
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
    nb_resultats = st.slider("Nombre de résultats à afficher", 5, 100, 20)
    
    # Seuil de score minimum
    seuil_score = st.slider("Seuil de score minimum", 0, 300, 10)
    
    # Poids des colonnes
    st.subheader("Poids des colonnes")
    if df is not None and len(colonnes_recherche) > 0:
        poids_colonnes = {}
        for col in colonnes_recherche:
            default_weight = 1.0 if col == "Désignation 1" else (0.4 if col == "Désignation 2" else 0.2)
            poids_colonnes[col] = st.slider(f"Poids de {col}", 0.0, 1.0, default_weight, 0.1)

# Barre de recherche principale avec mise en cache des résultats
@st.cache_data
def effectuer_recherche(df, requete, colonnes, seuil_score, nb_resultats, poids_colonnes):
    """Fonction pour mettre en cache les résultats de recherche"""
    try:
        stopwords_list = charger_stopwords()
    except:
        stopwords_list = []
    
    requete_nettoyee = clean_text(requete, stopwords_list)
    
    # Recherche optimisée - sans le paramètre seuil_score qui n'est pas accepté
    resultats = recherche_textuelle(
        df,
        requete_nettoyee,
        colonnes=colonnes,
        poids=poids_colonnes
    )
    
    # Filtrer selon le seuil de score manuellement
    resultats = resultats[resultats['score_global'] >= seuil_score]
    
    # Limiter le nombre de résultats
    resultats = resultats.head(nb_resultats)
    
    return resultats, requete_nettoyee

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
        # Exécuter la recherche avec mise en cache
        resultats, requete_nettoyee = effectuer_recherche(
            df, 
            requete, 
            colonnes_recherche,
            seuil_score,
            nb_resultats,
            poids_colonnes if 'poids_colonnes' in locals() else None
        )
        
        # Affichage des résultats
        st.subheader(f"Résultats de recherche ({len(resultats)} trouvés)")
        
        if len(resultats) > 0:
            # Préparation des données pour le tableau interactif
            # Colonnes à afficher
            colonnes_affichage = ["Code Pièce"] + colonnes_recherche
            
            if "Famille" in resultats.columns:
                colonnes_affichage.append("Famille")
                
            if "Section" in resultats.columns:
                colonnes_affichage.append("Section")
            
            # if "Prix" in resultats.columns:
            #     colonnes_affichage.append("Prix")
            
            # Ajouter la colonne de score à l'affichage
            #colonnes_affichage.append("Score")
            
            # Créer un dataframe d'affichage avec les colonnes choisies
            df_affichage = resultats[colonnes_affichage].copy()
            
            # Renommer la colonne score_global en Score
            df_affichage["Score"] = resultats["score_global"]
            
            # Formater les prix
            # if "Prix" in df_affichage.columns:
            #     df_affichage["Prix"] = df_affichage["Prix"].apply(lambda x: f"{x:.2f} €" if pd.notna(x) else "")
            
            # Afficher le tableau avec des options interactives
            st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
            
            # Tableau dynamique avec Streamlit
            st.dataframe(
                df_affichage,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score de pertinence",
                        help="Score de pertinence de 0 à 100",
                        format="%.1f",
                        min_value=0,
                        max_value=400,
                    ),
                    "Code Pièce": st.column_config.TextColumn(
                        "Code Pièce",
                        help="Référence de la pièce",
                        width="medium",
                    ),
                    # "Prix": st.column_config.TextColumn(
                    #     "Prix",
                    #     help="Prix en euros",
                    #     width="small",
                    # ),
                },
                hide_index=True,
                width=None,  # Pleine largeur
                height=400,   # Hauteur fixe
                use_container_width=True,
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
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
                    
            # Ajouter la possibilité de filtrer davantage les résultats
            with st.expander("Filtres supplémentaires"):
                # Filtrer par Famille si disponible
                if "Famille" in resultats.columns:
                    familles_disponibles = sorted(resultats["Famille"].dropna().unique())
                    famille_selectionnee = st.multiselect(
                        "Filtrer par Famille", 
                        options=familles_disponibles,
                        placeholder="Choisir une ou plusieurs familles"
                    )
                    
                    if famille_selectionnee:
                        resultats_filtres = resultats[resultats["Famille"].isin(famille_selectionnee)]
                        st.dataframe(
                            resultats_filtres[colonnes_affichage],
                            column_config={
                                "Score": st.column_config.ProgressColumn(
                                    "Score de pertinence",
                                    help="Score de pertinence de 0 à 100",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=100,
                                )
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.markdown(
                            get_download_link(resultats_filtres, "resultats_filtres.csv", "Télécharger les résultats filtrés"),
                            unsafe_allow_html=True
                        )
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
       
    4. **Optimisations** :
       - Mise en cache des résultats
       - Indexation des données
       - Vectorisation des calculs
    """)

# Pied de page
# Ajouter un footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #f0f2f6;">
    <p>Développé avec python et Streamlit </p>
    <p style="font-size: 0.8rem; color: #888;">© 2025 IoD solutions • Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)