import pandas as pd
import numpy as np
import re
import unicodedata
from difflib import SequenceMatcher
import string
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Cache pour éviter de recharger les stopwords
@lru_cache(maxsize=1)
def charger_stopwords(chemin_fichier='data/stopwords.csv', encoding='utf-8'):
    """
    Charge la liste des mots vides depuis un fichier CSV avec mise en cache.
    """
    try:
        stopwords_df = pd.read_csv(chemin_fichier, header=None, encoding=encoding)
        stopwords_list = list(stopwords_df[0])
        return stopwords_list
    except Exception as e:
        print(f"Erreur lors du chargement des stopwords: {str(e)}")
        return ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'de', 'du', 'en', 'à', 'au', 'aux', 'avec']

# Version mise en cache de clean_text
@lru_cache(maxsize=1000)
def clean_text_cached(text):
    """
    Version avec mise en cache de la fonction clean_text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
        
    # 1. Enlever les accents
    text = unidecode(str(text))
    
    # 2. Tout passer en majuscules
    text = text.upper()
    
    # 3. Tokeniser en séparant les mots
    tokens = text.split()
    
    # 4. Reconstruire la chaîne nettoyée
    return " ".join(tokens)

def clean_text(text, stopwords_list=None):
    """
    Enlève les accents, passe en majuscules et retire les mots vides.
    
    Args:
        text (str): Texte à nettoyer
        stopwords_list (list, optional): Liste des mots vides à filtrer
        
    Returns:
        str: Texte nettoyé
    """
    # Utiliser la version mise en cache pour le nettoyage de base
    text = clean_text_cached(text)
    
    # Si une liste de stopwords est fournie, les filtrer
    if stopwords_list:
        tokens = text.split()
        tokens = [t for t in tokens if t.lower() not in stopwords_list]
        text = " ".join(tokens)
    
    return text

def enlever_mots_communs_designations(df, cols_designation=None):
    """
    Pour chaque ligne du DataFrame, on enlève les mots déjà utilisés 
    dans une colonne 'Désignation X' des colonnes suivantes.
    
    Exemple:
      - Si le mot "CLE" apparaît dans "Désignation 1",
        on le supprime de "Désignation 2" et "Désignation 3".
        
    Args:
        df: DataFrame Pandas contenant vos colonnes de désignation
        cols_designation: Liste des noms de colonnes de désignation à traiter 
                          (dans l'ordre où on veut enlever les doublons).
                          
    Returns:
        Le DataFrame modifié avec les doublons supprimés.
    """
    # Par défaut, on traite les colonnes Désignation 1, 2, 3 (si elles existent dans df)
    if cols_designation is None:
        # Vous pouvez ajuster la liste selon vos besoins
        cols_designation = ["Désignation 1", "Désignation 2", "Désignation 3"]
    
    # Vérifier que les colonnes requises existent
    cols_existantes = [col for col in cols_designation if col in df.columns]
    if not cols_existantes:
        return df  # Aucune colonne à traiter
    
    # Copier le DataFrame pour ne pas modifier l'original
    df_clean = df.copy()
    
    # Fonction pour extraire les mots d'un texte
    def extraire_mots(texte):
        if pd.isna(texte) or not isinstance(texte, str):
            return set()
        # Convertir en majuscules pour uniformiser
        texte = texte.upper()
        # Extraire les mots (séquences de lettres et chiffres)
        mots = set(re.findall(r'\b\w+\b', texte))
        return mots
    
    # Parcourir chaque ligne du DataFrame
    for idx, row in df_clean.iterrows():
        mots_utilises = set()  # Ensemble des mots déjà utilisés
        
        # Traiter chaque colonne dans l'ordre
        for i, col in enumerate(cols_existantes):
            # Extraire les mots de la colonne courante
            mots_col = extraire_mots(row[col])
            
            # Pour toutes les colonnes après la colonne courante
            for next_col in cols_existantes[i+1:]:
                # Extraire les mots de la colonne suivante
                mots_next = extraire_mots(row[next_col])
                
                # Filtrer les mots qui sont déjà dans mots_col
                mots_filtres = mots_next - mots_col
                
                # Reconstruire le texte filtré
                if mots_filtres:
                    mots_next_liste = re.findall(r'\b\w+\b', str(row[next_col]).upper())
                    mots_filtres_liste = [mot for mot in mots_next_liste if mot in mots_filtres]
                    df_clean.at[idx, next_col] = " ".join(mots_filtres_liste)
                else:
                    df_clean.at[idx, next_col] = ""
            
            # Ajouter les mots de cette colonne à l'ensemble des mots utilisés
            mots_utilises.update(mots_col)
    
    return df_clean

def nettoyer_designations(df):
    """
    Nettoie les colonnes Désignation 2 et Désignation 3 en retirant les mots qui apparaissent
    déjà dans Désignation 1 ou Désignation 2.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les colonnes 'Désignation 1', 'Désignation 2', 'Désignation 3'
        
    Returns:
        pandas.DataFrame: DataFrame avec les colonnes de désignation nettoyées
    """
    # Vérifier que les colonnes requises existent
    colonnes_requises = ['Désignation 1', 'Désignation 2', 'Désignation 3']
    if not all(col in df.columns for col in colonnes_requises):
        # Si toutes les colonnes requises n'existent pas, renvoyer le DataFrame original
        return df
    
    # Utiliser la fonction enlever_mots_communs_designations
    return enlever_mots_communs_designations(df, colonnes_requises)

# Fonction optimisée pour obtenir les variantes d'un mot
@lru_cache(maxsize=500)
def obtenir_variantes(mot):
    """
    Obtient les variantes possibles d'un mot (singulier/pluriel) avec mise en cache.
    
    Args:
        mot (str): Mot dont on veut obtenir les variantes
        
    Returns:
        set: Ensemble des variantes possibles du mot
    """
    variantes = {mot}
    
    # Règles pour les pluriels/singuliers en français
    if mot.endswith('s'):
        # Possible singulier sans le 's'
        variantes.add(mot[:-1])
    else:
        # Possible pluriel avec 's'
        variantes.add(mot + 's')
        
    # Règles pour les terminaisons en 'x'
    if mot.endswith('x'):
        # Possible singulier sans le 'x'
        variantes.add(mot[:-1])
    elif mot.endswith('al'):
        # Pluriel en 'aux'
        variantes.add(mot[:-2] + 'aux')
    elif mot.endswith('au') or mot.endswith('eu'):
        # Pluriel en 'x'
        variantes.add(mot + 'x')
    
    # Règles pour les terminaisons en 'ail'/'eil'
    if mot.endswith('ail'):
        variantes.add(mot[:-3] + 'aux')
    
    # Règles pour les féminins/masculins
    if mot.endswith('eur'):
        variantes.add(mot[:-3] + 'euse')
    elif mot.endswith('teur'):
        variantes.add(mot[:-4] + 'trice')
    elif mot.endswith('er'):
        variantes.add(mot[:-2] + 'ere')
    
    return variantes

def calculer_score_pertinence(texte, requete):
    """
    Calcule un score de pertinence entre un texte et une requête en prenant en compte
    les fautes d'orthographe, pluriels/singuliers et autres variations.
    
    Args:
        texte: Le texte à comparer
        requete: La requête de recherche
    
    Returns:
        float: Score de pertinence
    """
    # Vérifier que le texte est une chaîne de caractères valide
    if pd.isna(texte) or not isinstance(texte, str) or texte.upper() == "NAN" or not texte.strip():
        return 0
    
    # Fonction de normalisation
    def normaliser(chaine):
        if not isinstance(chaine, str):
            return ""
        # Conversion en minuscules
        chaine = chaine.lower()
        # Suppression des accents
        chaine = unidecode(chaine)
        # Suppression des caractères spéciaux et des chiffres
        chaine = re.sub(r'[^\w\s]', ' ', chaine)
        # Conversion des espaces multiples en espace simple
        chaine = re.sub(r'\s+', ' ', chaine).strip()
        return chaine
    
    # Fonction pour calculer la similarité entre deux chaînes
    def similarite(s1, s2):
        # Si l'une des chaînes est vide, retourner 0
        if not s1 or not s2:
            return 0
            
        # Utiliser SequenceMatcher pour obtenir un ratio de similarité
        ratio = SequenceMatcher(None, s1, s2).ratio()
        return ratio
    
    # Normaliser les textes
    texte_norm = normaliser(texte)
    requete_norm = normaliser(requete)
    
    # Extraire les mots
    mots_texte = texte_norm.split()
    mots_requete = requete_norm.split()
    
    score = 0
    
    # 1. Correspondance exacte avec toute la requête
    if requete_norm in texte_norm:
        score += 100
    
    # 2. Correspondance approximative avec toute la requête
    similarite_globale = similarite(requete_norm, texte_norm)
    score += similarite_globale * 50
    
    # 3. Vérifier chaque mot de la requête
    for mot_requete in mots_requete:
        if len(mot_requete) <= 2:  # Ignorer les mots très courts (articles, etc.)
            continue
            
        max_similarite = 0
        meilleur_match_position = -1
        
        # Obtenir les variantes possibles du mot (singulier/pluriel)
        variantes_requete = obtenir_variantes(mot_requete)
        
        # Vérifier chaque mot du texte pour trouver le meilleur match
        for i, mot_texte in enumerate(mots_texte):
            # Vérifier la correspondance exacte avec les variantes
            for var in variantes_requete:
                if var == mot_texte:
                    sim = 1.0
                    if sim > max_similarite:
                        max_similarite = sim
                        meilleur_match_position = i
                    break
            
            # Si pas de correspondance exacte, calculer la similarité
            if max_similarite < 1.0:
                sim = similarite(mot_requete, mot_texte)
                if sim > max_similarite and sim > 0.7:  # Seuil de similarité
                    max_similarite = sim
                    meilleur_match_position = i
        
        # Ajouter des points en fonction de la similarité
        if max_similarite > 0:
            # Points pour la longueur du mot (plus de points pour les mots longs)
            points_longueur = max(5, 10 * len(mot_requete) / 10)
            
            # Points pour la similarité
            points_similarite = points_longueur * max_similarite
            
            # Bonus si le mot est au début du texte
            if meilleur_match_position == 0:
                points_similarite += 20
            # Bonus dégressif selon la position
            elif meilleur_match_position > 0:
                points_similarite += max(0, 10 - meilleur_match_position)
                
            score += points_similarite
    
    # 4. Bonus pour la couverture des mots de la requête
    mots_couverts = 0
    for mot_requete in mots_requete:
        if len(mot_requete) <= 2:  # Ignorer les mots très courts
            continue
            
        # Vérifier si le mot ou une variante est présent dans le texte
        variantes_requete = obtenir_variantes(mot_requete)
        for var in variantes_requete:
            if var in mots_texte or any(similarite(var, mot) > 0.8 for mot in mots_texte):
                mots_couverts += 1
                break
    
    # Calculer le pourcentage de couverture et ajouter des points
    mots_significatifs = sum(1 for mot in mots_requete if len(mot) > 2)
    if mots_significatifs > 0:
        pourcentage_couverture = mots_couverts / mots_significatifs
        score += pourcentage_couverture * 60
    
    # 5. Bonus pour l'ordre des mots
    if len(mots_requete) > 1 and len(mots_texte) > 1:
        # Vérifier si les mots apparaissent dans le même ordre
        sequence_trouvee = False
        for i in range(len(mots_texte) - len(mots_requete) + 1):
            sequence_valide = True
            for j, mot_requete in enumerate(mots_requete):
                if len(mot_requete) <= 2:  # Ignorer les mots très courts
                    continue
                if not (mot_requete in mots_texte[i+j] or similarite(mot_requete, mots_texte[i+j]) > 0.7):
                    sequence_valide = False
                    break
            if sequence_valide:
                sequence_trouvee = True
                break
        
        if sequence_trouvee:
            score += 40
    
    return score

def recherche_textuelle(df, requete, colonnes=["Désignation 1", "Désignation 2", "Désignation 3"], poids=None):
    """
    Recherche dans un DataFrame des éléments correspondant à une requête textuelle
    et retourne les résultats triés par pertinence en considérant plusieurs colonnes.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        requete (str): Texte à rechercher
        colonnes (list, optional): Liste des colonnes à considérer pour la recherche. 
                                  Par défaut ["Désignation 1", "Désignation 2", "Désignation 3"]
        poids (dict, optional): Dictionnaire des poids pour chaque colonne. 
                               Par défaut {"Désignation 1": 1.0, "Désignation 2": 0.4, "Désignation 3": 0.2}
    
    Returns:
        pandas.DataFrame: DataFrame filtré et trié par score de pertinence global
    """
    # Vérifier que le DataFrame contient les colonnes spécifiées
    colonnes_existantes = [col for col in colonnes if col in df.columns]
    if not colonnes_existantes:
        raise ValueError(f"Aucune des colonnes {colonnes} n'existe dans le DataFrame")
    
    # Poids par défaut si non spécifiés
    if poids is None:
        poids = {"Désignation 1": 1.0, "Désignation 2": 0.4, "Désignation 3": 0.2}
    
    # Copier le DataFrame pour éviter de modifier l'original
    df_resultat = df.copy()
    
    # Nettoyer les désignations pour éviter les redondances
    df_resultat = nettoyer_designations(df_resultat)
    
    # Calculer les scores pour chaque colonne
    for colonne in colonnes_existantes:
        df_resultat[f'score_{colonne}'] = df_resultat.apply(
            lambda row: calculer_score_pertinence(row[colonne], requete),
            axis=1
        )
    
    # Calculer le score global avec pondération
    df_resultat['score_global'] = 0
    for colonne in colonnes_existantes:
        # Utiliser le poids par défaut si la colonne n'est pas dans le dictionnaire des poids
        poids_colonne = poids.get(colonne, 0.2)
        df_resultat['score_global'] += df_resultat[f'score_{colonne}'] * poids_colonne
    
    # Créer des colonnes "len_xxx" pour compter le nombre de mots (pour trier)
    for col in colonnes_existantes:
        df_resultat[f'len_{col}'] = df_resultat[col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    # Trier par score global décroissant
    df_resultat = df_resultat.sort_values(by=['score_global'], ascending=False)
    
    return df_resultat

def clean_text2(text):
    """
    Enlève les accents, passe en majuscules et retire les mots vides.
    """
    return clean_text(text, charger_stopwords())

def filtrer_codes_piece_superieurs(dataframe, valeur_seuil=5000):
    """
    Filtre les codes pièce dont le numéro après 'UN' est supérieur à la valeur seuil.
    
    Args:
        dataframe: DataFrame contenant une colonne 'Code Pièce'
        valeur_seuil: Valeur numérique seuil (par défaut 5000)
        
    Returns:
        DataFrame filtré
    """
    # Créer une copie du DataFrame
    df_filtre = dataframe.copy()
    
    # Fonction pour extraire la partie numérique du code pièce
    def extraire_numero_un(code):
        try:
            # Rechercher le motif UN.XXXX ou UN.XXXX-YY
            match = re.search(r'UN\.?(\d+)', str(code), re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 0
        except:
            return 0
    
    # Appliquer la fonction d'extraction à la colonne 'Code Pièce'
    df_filtre['Numero_UN'] = df_filtre['Code Pièce'].apply(extraire_numero_un)
    
    # Filtrer les lignes dont le numéro est supérieur au seuil
    resultat = df_filtre[df_filtre['Numero_UN'] > valeur_seuil]
    
    # Supprimer la colonne temporaire
    resultat = resultat.drop(columns=['Numero_UN'])
    
    return resultat

def charger_fichier(chemin_fichier, encoding='utf-8'):
    """
    Charge un fichier CSV dans un DataFrame pandas.
    
    Args:
        chemin_fichier (str): Chemin du fichier à charger
        encoding (str): Encodage du fichier (par défaut: utf-8)
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données du fichier
    """
    try:
        df = pd.read_csv(chemin_fichier, encoding=encoding)
        df = filtrer_codes_piece_superieurs(df)
        
        # Sélectionner les colonnes importantes
        colonnes = ['Code Pièce', 'Désignation 1', 'Désignation 2', 'Désignation 3', 'Famille', 'Section', 'Prix', 'Prix ref. article']
        colonnes_existantes = [col for col in colonnes if col in df.columns]
        df = df[colonnes_existantes]
        
        # Prétraiter les colonnes textuelles
        for col in df.columns:
            if "Désignation" in col:
                df[col] = df[col].apply(clean_text2)
        
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {str(e)}")
        return None