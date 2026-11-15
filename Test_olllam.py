import time
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 1. Le LLM : temperature=0 pour des réponses stables (important pour l'évaluation)
model = "qwen2.5:7b"
llm = ChatOllama(model=model, temperature=0)

debut = time.time()
print("type de LLM :", model)
reponse = llm.invoke("Qu'est-ce qu'une franchise en assurance habitation ? Une phrase.")
print("LLM :", reponse.content)
print(f"Latence : {time.time() - debut:.1f} s")

# 2. Les embeddings : un texte devient un vecteur de nombres
model_emb = "bge-m3"
emb = OllamaEmbeddings(model=model_emb)
print("type de l'embedding :", model_emb)
vecteur = emb.embed_query("vol de vélo dans la cave")
print("Taille du vecteur :", len(vecteur))
print("5 premières valeurs :", vecteur[:5])