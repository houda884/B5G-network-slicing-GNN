# B5G Network Slicing — prédiction de QoS par slice (baseline, ML, GNN, événements)

## 1. Installation (Windows 10, Python 3.10.9)

```bat
:: dans le dossier du projet (celui qui contient src\ et requirements.txt)
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
`pip install torch` installe la version CPU sous Windows (suffisant ici). Vérification :
```bat
python -c "import torch, torch_geometric, networkx; print(torch.__version__, torch_geometric.__version__)"
```

## 2. Données
Le zip fourni contient un second zip (`slicing-sample.zip`). Dézipper **les deux** dans `data\raw\` :
```
data\raw\...\2.2.1-10kprocessed\
        graphs\  routings\  slices\  results_slicing_WFQ_1000_routing_*.tar.gz  (33 fichiers)
```
(la profondeur des sous-dossiers n'a pas d'importance : `datanetAPI` cherche le dossier qui contient
`graphs/`, `routings/` et `slices/`). **Ne pas décompresser les .tar.gz.**
Le fichier `src\datanetAPI.py` est votre API fournie (non modifiée).

## 3. Exécution
```bat
cd src
python run_all.py            :: tout enchaîner (≈ 40-60 min, l'étape GNN est la plus longue)
python run_all.py --quick    :: test de bon fonctionnement (GNN 8 epochs, 1 seed)
```
ou étape par étape :

| Script | Étape du guide | Livrable | Durée indicative |
|---|---|---|---|
| `01_extract.py` | préparation | `data/processed/*.csv, graphs.pkl` | 2-4 min |
| `02_eda.py` | Étape 1 | L1 : tableau des variables, qualité, figures | < 1 min |
| `03_events.py` | Étape 4 | L4 : événements | < 1 min |
| `04_baseline.py` | Étapes 2-3 | L3 : baselines + ML + ablation | 1-2 min |
| `05_gnn.py` | Étapes 5-6 | L5 : graphe + GNN (GCN/SAGE/GAT) + ablations | 30-50 min |
| `06_compare.py` | Étapes 7-8 | L6 : comparaison, robustesse, erreurs, importance | < 1 min |

Résultats : `results/tables/*.csv`, `results/figures/*.png`, journal GNN dans la console.

## 4. Formulation du problème (à défendre)
* **Unité** : un flux (src→dst) appartenant à un slice (eMBB / URLLC / mMTC), dans une simulation.
* **Cible** : `ln(délai moyen du flux)` (le délai va de 4 µs à 1825 s : on travaille en log).
  C'est la QoS que l'opérateur doit anticiper pour savoir si l'allocation courante (chemin + files WFQ)
  respecte le délai exigé par le slice (`delta`).
* **Entrées (jamais de sorties du simulateur)** : trafic, type de slice, SLA, chemin routé, capacités,
  configuration WFQ (part garantie par file), charge offerte.
* **Séparation** : par **simulation** (les flux d'une simulation partagent les liens → séparer par flux = fuite).
  `random` (même distribution) et `shift` (train = réseaux petits, test = les 35 % plus grands).
* **Événement** : flux dont ln(délai) est anormal pour son type de slice (z-score robuste MAD > 3) ou
  dont plus de 90 % des paquets sont perdus.
* **Modèles** : moyenne du train, Ridge (baseline), HistGradientBoosting (ML + features agrégées),
  GNN sur graphe flux–file–lien. Métriques : MAE, MSE (en ln-délai), globales et sur les événements.

## 5. Pièges découverts dans les données (à mentionner dans le rapport)
* Les simulations sont **indépendantes** (165 topologies différentes, une seule intensité de trafic,
  pas de série temporelle) : un « changement brusque » ne peut être qu'un changement de régime d'un flux
  ou un changement de distribution entre simulations (taille du réseau).
* Le chemin des slices (`slices_*.json`) et les attributs `traffic`/`reservation` des liens correspondent à un
  chemin **planifié** ; le chemin réellement simulé est celui de `routing_*.txt` (corrélation avec l'utilisation
  réelle des liens : 0.93-0.98 contre 0.4-0.7). Le projet utilise le routage.
* `PktsDrop`, `AvgDelay`, `linkUsage` sont des **sorties** : ne jamais les utiliser comme features.
* Les poids WFQ sont donnés **par port de sortie** (un groupe par port) et la correspondance ToS→file par noeud.
