# Moteur de Recherche de Pièces

## Présentation
Ce projet est une application web Streamlit implémentant un moteur de recherche intelligent pour un catalogue de pièces techniques. Le moteur permet aux utilisateurs de rechercher des produits en utilisant des descriptions libres, et utilise un algorithme optimisé pour trouver les correspondances les plus pertinentes même en présence de fautes d'orthographe ou de variations linguistiques.

## Structure du projet
- `app_opt.py` : Application Streamlit principale avec l'interface utilisateur
- `moteur_recherche_optimise.py` : Bibliothèque contenant le moteur de recherche et les algorithmes
- `data/` : Répertoire contenant les données
  - `stopwords.csv` : Liste des mots vides à filtrer
  - `data_filtre_v2.csv` : Fichier de données par défaut (facultatif)

## Installation

### Prérequis
- Python 3.11.1
- pip (gestionnaire de paquets Python)

### Installation des dépendances
```bash
pip install streamlit pandas numpy unidecode scikit-learn
```

### Lancement de l'application
```bash
streamlit run app_opt.py
```

## Format des données
L'application est conçue pour traiter des fichiers CSV comme "11022025_Clipper_extract.csv". Le fichier d'entrée doit contenir les colonnes suivantes :

- `Code Pièce` : Identifiant unique de la pièce
- `Désignation 1` : Description principale du produit
- `Désignation 2` : Description secondaire
- `Désignation 3` : Description tertiaire
- `Famille` : Catégorie de produit
- `Section` : Sous-catégorie
- `Prix` : Prix de vente
- `Prix ref. article` : Prix de référence

Un échantillon de données contient 3788 lignes avec exactement cette structure. Le moteur de recherche est optimisé pour traiter efficacement des fichiers de cette taille.

## Fonctionnement

### Moteur de recherche (`moteur_recherche_optimise.py`)

#### Traitement du texte
1. **Nettoyage des données**
   - Conversion en majuscules
   - Suppression des accents
   - Retrait des mots vides (stopwords)
   - Tokenisation

2. **Prétraitement des désignations**
   - Élimination des redondances entre `Désignation 1`, `Désignation 2` et `Désignation 3`
   - Les mots déjà présents dans une désignation ne sont pas répétés dans les suivantes

3. **Gestion des variations linguistiques**
   - Traitement des singuliers/pluriels
   - Variantes féminines/masculines
   - Tolérances aux fautes d'orthographe

#### Algorithme de recherche
L'algorithme principal `recherche_textuelle()` fonctionne comme suit :

1. **Calcul des scores par colonne**
   - Chaque désignation reçoit un score individuel
   - Le score prend en compte les correspondances exactes, les variations linguistiques, et les fautes de frappe

2. **Pondération des colonnes**
   - `Désignation 1` a un poids par défaut de 1.0
   - `Désignation 2` a un poids par défaut de 0.4
   - `Désignation 3` a un poids par défaut de 0.2
   - Ces poids sont paramétrables

3. **Métriques de pertinence**
   - Bonus pour les correspondances exactes (+100 points)
   - Bonus pour les mots trouvés en début de texte (+20 points)
   - Bonus pour les mots longs (5-10 points selon la longueur)
   - Bonus pour l'ordre des mots (+40 points)
   - Bonus pour le taux de couverture des mots de la requête (jusqu'à +60 points)

4. **Techniques d'optimisation**
   - Mise en cache des opérations coûteuses
   - Vectorisation des calculs
   - Pré-filtrage des désignations

### Interface utilisateur (`app_opt.py`)

1. **Configuration**
   - Chargement de fichier CSV par glisser-déposer
   - Configuration des colonnes à rechercher
   - Réglage des poids de chaque colonne
   - Ajustement du seuil de score minimal
   - Définition du nombre de résultats à afficher

2. **Affichage des résultats**
   - Tableau interactif avec colonnes configurables
   - Indicateurs visuels des scores
   - Options de téléchargement des résultats
   - Filtres supplémentaires (par famille, etc.)
   - Vue détaillée du meilleur résultat

3. **Optimisations UI**
   - Mise en cache des résultats
   - Affichage progressif des résultats
   - Métriques de statistiques du jeu de données

## Exemples d'utilisation

### Recherche simple
```
clé à oeil 17mm
```

### Recherche avec variations
```
cles a choc pneumatiques
```

### Recherche avec orthographe approximative
```
tournvis cruciforme
```

## Personnalisation

### Ajout de stopwords
Vous pouvez modifier la liste des mots vides en éditant le fichier `data/stopwords.csv`.

### Modification des poids
Dans l'interface, ajustez les curseurs de poids pour personnaliser l'importance de chaque colonne.

### Filtrage avancé
Utilisez la section "Filtres supplémentaires" pour affiner les résultats par famille ou autres critères.

## Performances
- Le moteur utilise une mise en cache intelligente pour optimiser les performances.
- Sur un jeu de données d'environ 3800 lignes, le temps de réponse moyen est inférieur à 1 seconde.
- L'algorithme de scoring est optimisé pour privilégier la précision sans sacrifier la tolérance aux variations.
- Le moteur a été testé et optimisé pour Python 3.11.1.

## Limitations connues
- Les fichiers CSV très volumineux (>100 000 lignes) peuvent entraîner un ralentissement.
- La mise en cache consomme de la mémoire, ce qui peut poser problème sur des machines avec peu de RAM.
- L'application filtre automatiquement les codes pièce dont le numéro après "UN" est supérieur à 5000.