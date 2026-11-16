# RAG Assurance

Assistant qui répond aux questions d'un assuré sur son contrat d'assurance habitation, en citant ses sources.
L'utilisateur dépose ses Conditions Particulières ; le système retrouve les Conditions Générales de son contrat et y cherche la réponse. 100 % local (Ollama).

## Avancement

- [x] Corpus : 11 PDF (CG + IPID), 4 assureurs, 25 questions de référence
- [x] Indexation : chunks + embeddings bge-m3 + Chroma
- [x] Extraction des CP et identification du contrat
- [x] Recherche filtrée par contrat + génération sourcée
- [x] Évaluation de la recherche et de la génération
- [x] Traçage LangSmith
- [ ] Recherche hybride (BM25) + reranking
- [ ] Démo Streamlit

## Résultats

| Mesure | Résultat |
|---|---|
| Identification du contrat | 6/6 |
| Recherche : Hit@5 / Recall@5 | 81 % / 51 % |
| Génération : refus à tort | 0 % |
| Latence par réponse (qwen3:8b) | ~16 s |

## Lancer

```bash
ollama pull qwen2.5:7b && ollama pull qwen3:8b && ollama pull bge-m3
pip install -r requirements.txt
python -m src.indexing.indexing
python -m src.generation.generation data/cp/cp_01_luko_etendue.pdf "Suis-je couvert si on me vole mon vélo dans ma cave ?"
```