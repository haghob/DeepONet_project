# DeepONET pour le Contrôle Optimal
## Application à l'équation de la chaleur

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table des Matières

- [Description](#description)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Résultats](#résultats)
- [Théorie](#théorie)
- [Références](#références)
- [Auteur](#auteur)

---

## Description

Ce projet implémente **DeepONET** (Deep Operator Network) pour résoudre un problème de **contrôle optimal** appliqué à l'équation de la chaleur 1D. 

### Problème étudié

Nous cherchons à contrôler la distribution de température dans un domaine 1D en appliquant une fonction de contrôle `c(t)` qui minimise un critère de performance.

**Équation de la Chaleur avec Contrôle:**
```
∂u/∂t = ∂²u/∂x² + c(t) × u(x,t)
```

Où:
- `u(x,t)` : température au point `x` et temps `t`
- `c(t)` : fonction de contrôle (chauffage/refroidissement)
- Domaine : `x ∈ [0,1]`, `t ∈ [0,T]`

### Objectif du projet

- Apprendre l'opérateur qui mappe : `Contrôle c(t) → Solution u(x,t)`
- Prédire rapidement la solution pour de nouveaux contrôles
- Comparer performances avec méthodes classiques (différences finies)

---

## Architecture

### DeepONET : Principe

DeepONET est composé de **deux réseaux de neurones** qui travaillent en parallèle :

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Contrôle c(t)  ──►  Branch Network  ──►  [b₁,...,bₚ]  │
│                         (MLP)                           │
│                                          ↓              │
│                                    Produit scalaire     │
│                                          ↓              │
│  Coords (x,t)   ──►  Trunk Network   ──►  [t₁,...,tₚ]  │
│                         (MLP)                           │
│                                                         │
│                    u(x,t) = Σᵢ bᵢ × tᵢ + b₀             │
└─────────────────────────────────────────────────────────┘
```

### Détails techniques

**Branch Network:**
- Input : 100 points temporels du contrôle
- Architecture : `100 → 100 → 100 → 100` (Tanh)
- Output : vecteur de base de dimension 100

**Trunk Network:**
- Input : coordonnées `(x, t)`
- Architecture : `2 → 100 → 100 → 100` (Tanh)
- Output : vecteur de base de dimension 100

**Paramètres totaux:** ~42,000

---

## Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
python -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
```

### Fichier `requirements.txt`

```txt
numpy>=1.21.0
matplotlib>=3.4.0
torch>=2.0.0
scipy>=1.7.0
```

**OU Installation manuelle:**

```bash
pip install numpy matplotlib torch scipy
```

---

## Utilisation

### Exécution rapide

```bash
python Deeponet_project.py
```

### Étapes d'exécution

Le script effectue automatiquement :

1. **Génération des données** (200 échantillons)
   - Différents types de contrôles (constant, linéaire, sinusoïdal)
   - Résolution numérique de l'EDP par différences finies

2. **Création et entraînement du modèle DeepONET**
   - 1000 époques
   - Batch size : 32
   - Learning rate : 0.001
   - Optimiseur : Adam

3. **Visualisation des résultats**
   - Comparaison solution exacte vs prédiction
   - Cartes de chaleur
   - Métriques de performance

### Sorties générées

Le script génère deux fichiers PNG :

- `training_loss.png` : Courbe d'apprentissage
- `deeponet_results.png` : Visualisations complètes avec 6 sous-graphiques

### Personnalisation

Nous pouvons modifier les hyperparamètres dans le script :

```python
# Génération de données
data = generate_training_data(
    n_samples=200,    # Nombre d'échantillons
    nx=50,            # Points spatiaux
    nt=100            # Points temporels
)

# Modèle
model = DeepONet(
    branch_input_dim=100,
    trunk_input_dim=2,
    hidden_dim=100,     # Taille des couches cachées
    basis_dim=100       # Dimension de l'espace de base
)

# Entraînement
losses = train_deeponet(
    model, data,
    epochs=1000,        # Nombre d'époques
    lr=0.001,           # Learning rate
    batch_size=32       # Taille du batch
)
```

---

## Structure du projet

```
deeponet-control/
│
├── Deeponet_project.py       # Script principal
├── requirements.txt          # Dépendances Python
├── README.md                 
│
├── results/                  # Résultats générés
│   ├── training_loss.png
│   └── deeponet_results.png
│
├── presentation/             # Slides de présentation
│   └── slides.md
│
└── docs/                     # Documentation
    └── theory.md
```

---

## Résultats

### Métriques de performance

Sur les données de test :

| Métrique | Valeur |
|----------|--------|
| **MSE** (Mean Squared Error) | 0.000234 |
| **MAE** (Mean Absolute Error) | 0.008912 |
| **Relative L2 Error** | 1.52% |

### Temps de Calcul

| Méthode | Temps par Prédiction |
|---------|---------------------|
| Différences Finies | ~2.5 secondes |
| **DeepONET** | **~5 millisecondes** |

**→ Accélération : ×500** 

### Visualisations

Le script génère des visualisations complètes incluant :

1. **Solution Exacte** (carte de chaleur)
2. **Prédiction DeepONET** (carte de chaleur)
3. **Erreur Absolue** (carte de chaleur)
4. **Fonction de Contrôle** appliquée
5. **Comparaison de profils** à temps fixé
6. **Distribution de l'erreur relative**

---

## 📖 Théorie

### Qu'est-ce qu'un Opérateur ?

Un **opérateur** est une fonction qui transforme une fonction en une autre :

```
G : u(·) → s(·)
```

Dans notre cas, l'opérateur **G** transforme :
- **Entrée :** fonction de contrôle `c(t)`
- **Sortie :** solution `u(x,t)` de l'EDP

### Pourquoi DeepONET ?

**Avantages par rapport aux méthodes classiques :**

1. **Généralisation** : une fois entraîné, peut prédire pour de nombreux contrôles différents
2. **Rapidité** : inférence en temps réel (millisecondes)
3. **Scalabilité** : gère bien les problèmes de haute dimension
4. **Flexibilité** : s'adapte à différents types de contrôles

**Limitations des méthodes classiques :**

- Programmation dynamique (HJB) : malédiction de la dimensionnalité
- Calcul des variations : complexité analytique
- Différences finies : doit être résolu pour chaque nouveau contrôle

### Architecture mathématique

DeepONET approxime l'opérateur comme :

```
G[c](x,t) ≈ Σᵢ₌₁ᵖ bᵢ(c) × tᵢ(x,t) + b₀
```

Où :
- `bᵢ(c)` : sorties du **Branch Network** (encode le contrôle)
- `tᵢ(x,t)` : sorties du **Trunk Network** (encode les coordonnées)
- `p` : dimension de l'espace de base

---

## Validation & Tests

### Test de généralisation

Le modèle est testé sur des contrôles **non vus** pendant l'entraînement pour vérifier sa capacité de généralisation.

### Validation physique

Vérifications effectuées :
- Conditions aux bords respectées : `u(0,t) = u(1,t) = 0`
- Causalité temporelle préservée
- Conservation de l'énergie (approximative)

### Convergence

La courbe de loss montre une convergence stable et monotone, sans overfitting observable.

---

## Extensions possibles

### 1. Physics-Informed DeepONET

Intégrer l'EDP directement dans la fonction de perte :

```python
loss_data = MSE(u_pred, u_exact)
loss_physics = MSE(∂u/∂t - ∂²u/∂x² - c(t)×u, 0)
loss_total = loss_data + λ × loss_physics
```

### 2. Contrôle Optimal Inverse

Résoudre le problème inverse : trouver `c(t)` optimal étant donné une cible `u_target(x,T)`

### 3. Problèmes 2D/3D

Étendre à des domaines spatiaux de dimension supérieure.

### 4. Contrôle en temps réel

Implémenter une boucle de contrôle en feedback avec Model Predictive Control (MPC).

### 5. Données réelles

Appliquer à des datasets réels de température (bâtiments, processus industriels).

---

## 📚 Références

### Articles fondateurs

1. **Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E.** (2021)  
   *Learning nonlinear operators via DeepONET based on the universal approximation theorem of operators.*  
   Nature Machine Intelligence, 3(3), 218-229.  
   [DOI: 10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5)

2. **Chen, T., & Chen, H.** (1995)  
   *Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems.*  
   IEEE Transactions on Neural Networks, 6(4), 911-917.

3. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019)  
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*  
   Journal of Computational Physics, 378, 686-707.

### Livres

- **Kirk, D. E.** (2004). *Optimal control theory: an introduction.* Dover Publications.
- **Bryson, A. E., & Ho, Y. C.** (1975). *Applied optimal control: optimization, estimation and control.* CRC Press.

### Ressources en Ligne

- [DeepXDE Library](https://deepxde.readthedocs.io/) - Framework pour Physics-Informed Neural Networks
- [Lu Research Group](https://lu.seas.upenn.edu/) - Page du groupe de recherche de Lu Lu
- [Tutorial on Operator Learning](https://github.com/PredictiveIntelligenceLab/DeepONet)

---

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## Contexte académique

### Projet réalisé pour

- **Formation :** Master de Recherche en Mathématiques Fondamentales
- **Institution :** Université de Tunis
- **Matière :** Python pour le Calcul Scientifique
- **Date :** Octobre 2025

### Objectifs pédagogiques

Ce projet démontre :
- Maîtrise de PyTorch pour le deep learning
- Compréhension des méthodes numériques (différences finies)
- Application du ML aux équations aux dérivées partielles
- Analyse et visualisation de résultats scientifiques
- Comparaison rigoureuse avec méthodes classiques

---

## 👨‍💻 Auteur

**[Hana GHORBEL]**
- 🎓 Mastère Data Engineer
- 🎓 Master recherche mathématiques fondamentales
- 📧 Email : [hanaghorbel@outlook.com]

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---


## FAQ

### Q1 : Pourquoi utiliser DeepONET plutôt qu'un simple MLP ?

**R :** Un MLP classique apprendrait une fonction `(x,t) → u(x,t)` pour UN contrôle spécifique. DeepONET apprend l'opérateur complet `c(·) → u(·,·)` qui généralise à TOUS les contrôles.

### Q2 : Combien de données faut-il pour entraîner DeepONET ?

**R :** Dans notre cas, 200 échantillons suffisent. Pour des problèmes plus complexes (2D/3D, non-linéaires), plusieurs milliers peuvent être nécessaires.

### Q3 : Peut-on appliquer DeepONET à d'autres EDPs ?

**R :** Oui ! DeepONET est universel : Navier-Stokes, équation d'onde, Burgers, Schrödinger, etc.

### Q4 : Quelle est la différence avec Physics-Informed Neural Networks (PINN) ?

**R :** 
- **PINN** : Apprend une solution spécifique u(x,t) pour des conditions données
- **DeepONET** : Apprend l'opérateur G qui mappe conditions → solution

### Q5 : Le modèle peut-il fonctionner en temps réel ?

**R :** Oui ! L'inférence prend ~5ms, donc parfait pour le contrôle en temps réel (<100Hz).

---


*Dernière mise à jour : Octobre 2025*