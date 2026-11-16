# 🛡️ RAG Assurance — Assistant de contrat d'assurance habitation

Assistant qui répond aux questions d'un assuré **sur son propre contrat**, en citant ses sources.
L'utilisateur dépose ses **Conditions Particulières (CP)** ; le système identifie automatiquement les **Conditions Générales (CG)** de son contrat, y cherche les passages utiles et génère une réponse sourcée.

**100 % local** : LLM, embeddings et reranker tournent sur la machine (Ollama + PyTorch). Aucune donnée ne sort.

![Démo](docs/demo.png)

---

## Fonctionnalités

- **Identification du contrat** : à partir des CP, le système retrouve l'assureur, la référence et l'édition des CG applicables.
- **Recherche filtrée par contrat** : uniquement dans les CG du client (bon assureur, bonne version), puis **reranking** par cross-encoder.
- **Réponses sourcées** : chaque information renvoie à la CP ou à une page précise des CG.
- **Refus maîtrisé** : si les CG du contrat ne sont pas dans la base, le système le dit au lieu de répondre avec un autre contrat.
- **Traçage** : chaque question est tracée dans LangSmith (recherche, reranking, génération).

## Architecture

```
Dépôt des CP ──► LLM (sortie structurée) ──► fiche : assureur, référence, édition
                                                 │
                                    règles ──► contrat_id   (aucun ? → message de refus)
                                                 │
Question ──► recherche vectorielle filtrée (20) ──► reranking (5) ──► LLM ──► réponse + sources
```

## Stack

Python · LangChain · Ollama (qwen2.5:7b, qwen3:8b, bge-m3) · Chroma · sentence-transformers (bge-reranker-v2-m3) · LangSmith · Streamlit

## Résultats

Évaluation sur 25 questions de référence vérifiées (16 pour la recherche).

| Mesure | Résultat |
|---|---|
| Identification du contrat (6 CP) | 6/6 |
| Identification sur cas pièges (versions proches ou absentes) — règles vs LLM | 5/5 vs 1/5 |
| Recherche v1 (vecteurs) — Hit@5 / Recall@5 | 81 % / 51 % |
| Recherche v2 (vecteurs + reranking) — Hit@5 / Recall@5 | **88 % / 50 %** |
| Recherche hybride BM25 + vecteurs | aucun gain → non retenue |
| Génération — refus à tort | 0 % |
| Latence : recherche / génération | ~0,7 s / ~15 s |

## Choix techniques

- **Les CP ne sont jamais indexées** : elles sont courtes et personnelles. Elles sont lues en entier dans le prompt, et ne vont pas dans la base vectorielle partagée.
- **Le contrat est choisi par des règles, pas par le LLM** : sur les cas pièges, un LLM choisit la version la plus proche au lieu de refuser.
- **Les sources sont ajoutées par le code** à partir des numéros cités par le LLM : le fichier et la page sont toujours exacts.
- **Chaque amélioration est mesurée** avant d'être gardée (hybride abandonné, reranking retenu).

## Limites

- Le modèle local (8B) peut conclure trop vite ou refuser à tort sur les questions complexes.
- Le jeu d'évaluation est petit (16 à 25 questions) : les écarts d'une question restent à confirmer.
- L'extraction PDF perd certains caractères (tirets des références légales) et aplatit les tableaux.

## Installation

Prérequis : Python 3.11+, [Ollama](https://ollama.com), un GPU NVIDIA recommandé.

```bash
git clone <url-du-repo>
cd rag-assurance
pip install -r requirements.txt

ollama pull qwen2.5:7b
ollama pull qwen3:8b
ollama pull bge-m3
```

Traçage LangSmith (optionnel) : créer un fichier `.env` à la racine :
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=rag-assurance
```

## Utilisation

```bash
python -m src.indexing.indexing     # indexation des CG (une fois)
streamlit run app.py                # lancer la démo
```

Des CP fictives sont fournies dans `data/cp/` pour tester (dont une d'un assureur absent de la base).

## Structure

```
rag-assurance/
├── config.py            # modèles et paramètres
├── app.py               # démo Streamlit
├── data/
│   ├── cg/              # CG + IPID (PDF) + catalogue.json
│   ├── cp/              # CP fictives
│   └── eval/            # questions de référence
├── src/
│   ├── indexing/        # PDF → chunks → Chroma
│   ├── extraction_cp/   # fiche CP + identification du contrat
│   ├── retrieval/       # recherche filtrée + reranking
│   ├── generation/      # réponse sourcée
│   └── evaluation/      # évaluation de la recherche et de la génération
└── results/             # résultats des évaluations
```

## Données

Les CG et IPID sont des documents publics publiés par les assureurs (Luko, Société Générale, GMF, Matmut). Les CP sont **fictives**, créées pour le projet.

---

**Omar El Moujahid** 