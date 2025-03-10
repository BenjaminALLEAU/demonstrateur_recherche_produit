import pandas as pd
import numpy as np
import re
import unicodedata
from difflib import SequenceMatcher
import string
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text, stopwords_list=None):
    """
    Enlève les accents, passe en majuscules et retire les mots vides.
    
    Args:
        text (str): Texte à nettoyer
        stopwords_list (list, optional): Liste des mots vides à filtrer
        
    Returns:
        str: Texte nettoyé
    """
    # Convertir en chaîne de caractères (au cas où certaines cases seraient NaN)
    text = str(text)
    
    # 1. Enlever les accents
    text = unidecode(text)
    
    # 2. Tout passer en majuscules
    text = text.upper()
    
    # 3. Tokeniser en séparant les mots
    tokens = text.split()
    
    # 4. Retirer les mots vides si une liste est fournie
    if stopwords_list:
        tokens = [t for t in tokens if t.lower() not in stopwords_list]
    
    # 5. Reconstruire la chaîne nettoyée
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
    
    def remove_duplicate_words_in_row(row):
        # Ensemble des mots déjà rencontrés dans les colonnes précédentes
        used_words = set()
        
        for col in cols_designation:
            if col in row:
                # Découper la désignation en mots
                words = row[col].split()
                
                # Garder uniquement les mots qui n'ont pas encore été vus
                new_words = [w for w in words if w not in used_words]
                
                # Mettre à jour la colonne avec les mots filtrés
                row[col] = " ".join(new_words)
                
                # Ajouter ces nouveaux mots à l'ensemble de mots utilisés
                used_words.update(new_words)
        
        return row
    
    # Appliquer la fonction à chaque ligne du DataFrame
    df = df.apply(remove_duplicate_words_in_row, axis=1)
    return df

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
    for col in colonnes_requises:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame")
    
    # Copier le DataFrame pour ne pas modifier l'original
    df_clean = df.copy()
    
    # Fonction pour diviser un texte en mots
    def extraire_mots(texte):
        if pd.isna(texte) or not isinstance(texte, str) or texte.upper() == "NAN":
            return set()
        mots = set(re.findall(r'\b\w+\b', texte))
        return mots
    
    # Fonction pour supprimer les mots spécifiés d'un texte
    def supprimer_mots(texte, mots_a_supprimer):
        if pd.isna(texte) or not isinstance(texte, str) or texte.upper() == "NAN":
            return texte
        
        texte_norm = texte
        mots_texte = re.findall(r'\b\w+\b', texte_norm)
        
        # Construire un dictionnaire des positions des mots
        positions = {}
        for i, mot in enumerate(mots_texte):
            positions[mot] = positions.get(mot, []) + [i]
        
        # Trouver les positions des mots à supprimer
        indices_a_supprimer = []
        for mot in mots_a_supprimer:
            if mot in positions:
                indices_a_supprimer.extend(positions[mot])
        
        # Si aucun mot à supprimer, retourner le texte original
        if not indices_a_supprimer:
            return texte
        
        # Diviser le texte original en caractères pour préserver la casse
        chars = list(texte)
        
        # Déterminer les limites des mots dans le texte original
        limites = []
        mot_courant = ""
        debut_mot = None
        
        for i, char in enumerate(texte):
            if re.match(r'\w', char):
                if debut_mot is None:
                    debut_mot = i
                mot_courant += char.lower()
            else:
                if debut_mot is not None:
                    limites.append((debut_mot, i, mot_courant))
                    mot_courant = ""
                    debut_mot = None
        
        # Ajouter le dernier mot si nécessaire
        if debut_mot is not None:
            limites.append((debut_mot, len(texte), mot_courant))
        
        # Identifier les limites des mots à supprimer
        segments_a_supprimer = []
        for i, (debut, fin, mot) in enumerate(limites):
            if mot in mots_a_supprimer:
                # Inclure l'espace suivant ou précédent dans le segment à supprimer
                debut_seg = debut
                fin_seg = fin
                
                # Étendre à l'espace suivant si ce n'est pas le dernier mot
                if fin < len(texte) and texte[fin].isspace():
                    fin_seg = fin + 1
                # Ou à l'espace précédent si ce n'est pas le premier mot
                elif debut > 0 and texte[debut-1].isspace():
                    debut_seg = debut - 1
                
                segments_a_supprimer.append((debut_seg, fin_seg))
        
        # Trier les segments dans l'ordre inverse pour ne pas perturber les indices
        segments_a_supprimer.sort(reverse=True)
        
        # Construire le texte résultant en supprimant les segments
        resultat = texte
        for debut, fin in segments_a_supprimer:
            resultat = resultat[:debut] + resultat[fin:]
        
        # Supprimer les espaces multiples
        resultat = re.sub(r'\s+', ' ', resultat).strip()
        
        return resultat
    
    # Traiter chaque ligne
    for idx, row in df_clean.iterrows():
        # Extraire les mots de Désignation 1
        mots_design1 = extraire_mots(row['Désignation 1'])
        
        # Nettoyer Désignation 2 en supprimant les mots présents dans Désignation 1
        if not pd.isna(row['Désignation 2']) and isinstance(row['Désignation 2'], str):
            df_clean.at[idx, 'Désignation 2'] = supprimer_mots(row['Désignation 2'], mots_design1)
        
        # Extraire les mots de Désignation 2
        mots_design2 = extraire_mots(row['Désignation 2'])
        
        # Nettoyer Désignation 3 en supprimant les mots présents dans Désignation 1 et 2
        if not pd.isna(row['Désignation 3']) and isinstance(row['Désignation 3'], str):
            df_clean.at[idx, 'Désignation 3'] = supprimer_mots(
                row['Désignation 3'], 
                mots_design1.union(mots_design2)
            )
    
    # Enlever les mots communs 
    df_clean = enlever_mots_communs_designations(df_clean)
    return df_clean

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
    if pd.isna(texte) or not isinstance(texte, str) or texte.upper() == "NAN":
        return 0
    
    # Fonction de normalisation améliorée
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
    
    # Fonction pour obtenir les variantes d'un mot (singulier/pluriel)
    def obtenir_variantes(mot):
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

def recherche_textuelle(df, requete, colonnes=["Désignation 1", "Désignation 2", "Désignation 3"]):
    """
    Recherche dans un DataFrame des éléments correspondant à une requête textuelle
    et retourne les résultats triés par pertinence en considérant plusieurs colonnes.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        requete (str): Texte à rechercher
        colonnes (list, optional): Liste des colonnes à considérer pour la recherche. 
                                  Par défaut ["Désignation 1", "Désignation 2", "Désignation 3"]
    
    Returns:
        pandas.DataFrame: DataFrame filtré et trié par score de pertinence global
    """
    # Vérifier que le DataFrame contient les colonnes spécifiées
    for colonne in colonnes:
        if colonne not in df.columns:
            raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame")
    
    # Copier le DataFrame pour éviter de modifier l'original
    df_resultat = nettoyer_designations(df.copy())
    
    # Calculer les scores pour chaque colonne
    for colonne in colonnes:
        df_resultat[f'score_{colonne}'] = df_resultat.apply(
            lambda row: calculer_score_pertinence(row[colonne], requete),
            axis=1
        )
    
    # Calculer le score global avec pondération (Désignation 1 a plus de poids)
    poids = {"Désignation 1": 1.0, "Désignation 2": 0.4, "Désignation 3": 0.2}
    
    # Initialiser le score global
    df_resultat['score_global'] = 0
    
    # Ajouter la contribution de chaque colonne au score global
    for colonne in colonnes:
        if colonne in poids:
            df_resultat['score_global'] += df_resultat[f'score_{colonne}'] * poids[colonne]
    
    # Filtrer les éléments avec un score global > 0
    df_resultat = df_resultat[df_resultat['score_global'] > 0]
    
    # Créer des colonnes "len_xxx" pour compter le nombre de mots
    cols_designation = ["Désignation 1", "Désignation 2", "Désignation 3"]
    for col in cols_designation:
        if col in df_resultat.columns:
            df_resultat[f'len_{col}'] = df_resultat[col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    # Trier par score global décroissant et par nombre de mots
    sort_cols = ["score_global"]
    df_resultat = df_resultat.sort_values(by=sort_cols, ascending=False)
    df_resultat
    
    return df_resultat

def clean_text2(text):
    """
    Enlève les accents, passe en majuscules et retire les mots vides.
    """
    # Convertir en chaîne de caractères (au cas où certaines cases seraient NaN)
    text = str(text)
    
    # 1. Enlever les accents
    text = unidecode(text)
    
    # 2. Tout passer en majuscules
    text = text.upper()
    
    # 3. Tokeniser en séparant les mots
    tokens = text.split()
    
    # 4. Retirer les mots vides
    tokens = [t for t in tokens if t.lower() not in charger_stopwords()]
    
    # 5. Reconstruire la chaîne nettoyée
    text = " ".join(tokens)
    
    return text

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
        df = df[['Code Pièce', 'Désignation 1', 'Désignation 2', 'Désignation 3','Famille','Section', 'Prix', 'Prix ref. article']]
        for col in df.columns:
            if "Désignation" in col:
                df[col] = df[col].apply(clean_text2)
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {str(e)}")
        return None

def charger_stopwords(chemin_fichier='data/stopwords.csv', encoding='utf-8'):
    """
    Charge la liste des mots vides depuis un fichier CSV.
    
    Args:
        chemin_fichier (str): Chemin du fichier contenant les mots vides
        encoding (str): Encodage du fichier (par défaut: utf-8)
        
    Returns:
        list: Liste des mots vides
    """
    try:
        stopwords_df = pd.read_csv(chemin_fichier, header=None, encoding=encoding)
        stopwords_list = list(stopwords_df[0])
        return stopwords_list
    except Exception as e:
        print(f"Erreur lors du chargement des stopwords: {str(e)}")
        # Liste de stopwords française par défaut
        return ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'de', 'du', 'en', 'à', 'au', 'aux', 'avec']
    

