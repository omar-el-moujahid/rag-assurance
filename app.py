"""Démo Streamlit : l'utilisateur dépose ses conditions particulières (CP), puis pose ses questions.
Lancement (depuis la racine du projet) :  streamlit run app.py
"""
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st
from langsmith import tracing_context

from src.extraction_cp.extraction import analyser_cp
from src.retrieval.retrieval import ouvrir_store, rechercher
from src.generation.generation import repondre            # <- A ADAPTER si ton module s'appelle autrement



# def generer(question, texte_cp, fiche, passages, contrat_id):
#     """Point de contact avec l'étape génération."""
#     return repondre(question, texte_cp, passages, contrat_id)



st.set_page_config(
    page_title="Assistant contrat Assurance",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def store():
    """La base Chroma est ouverte une seule fois pour toute l'application, pas à chaque question."""
    return ouvrir_store()


# --- Etat de la session : un identifiant = un thread LangSmith (toute la conversation regroupée)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.cp = None            # (nom_fichier, texte_cp, fiche, contrat_id)

trace = {"session_id": st.session_state.session_id}


# ------------------------------------------------------------------ Barre latérale : dépôt de la CP
with st.sidebar:
    st.header("1. Vos conditions particulières")
    fichier = st.file_uploader("Déposez votre CP (PDF)", type="pdf")

    if fichier and (st.session_state.cp is None or st.session_state.cp[0] != fichier.name):
        with st.spinner("Lecture de la CP et identification du contrat..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(fichier.getvalue())
            t0 = time.perf_counter()
            with tracing_context(metadata=trace):
                texte_cp, fiche, contrat_id = analyser_cp(Path(tmp.name))
            Path(tmp.name).unlink(missing_ok=True)
        st.session_state.cp = (fichier.name, texte_cp, fiche, contrat_id)
        st.session_state.messages = []
        st.caption(f"Analyse en {time.perf_counter() - t0:.1f} s")

    if st.session_state.cp:
        _, _, fiche, contrat_id = st.session_state.cp
        if contrat_id:
            st.success(f"Contrat reconnu : **{contrat_id}**")
        else:
            st.warning("Contrat non reconnu : il n'est pas dans la base. Je ne pourrai pas répondre.")
        with st.expander("Fiche extraite"):
            st.json(fiche if isinstance(fiche, dict) else {"fiche": str(fiche)})

    st.divider()
    afficher_passages = st.toggle("Afficher les passages retrouvés", value=True)
    if st.button("Nouvelle conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Thread LangSmith : `{st.session_state.session_id[:8]}`")


# ------------------------------------------------------------------ Zone principale : le chat
st.title("🛡️ Assistant contrat Assurance")

st.caption("Réponses fondées sur les conditions générales de votre assureur et sur vos CP, avec sources.")

if st.session_state.cp is None:
    st.info("Commencez par déposer vos conditions particulières dans le panneau de gauche.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("passages") and afficher_passages:
            with st.expander(f"{len(msg['passages'])} passages · {msg['infos']}"):
                for i, (source, extrait) in enumerate(msg["passages"], start=1):
                    st.markdown(f"**[{i}] {source}**")
                    st.text(extrait)

question = st.chat_input("Votre question sur votre contrat...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    _, texte_cp, fiche, contrat_id = st.session_state.cp
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les conditions générales et rédaction..."):
            with tracing_context(metadata=trace):
                t0 = time.perf_counter()
                resultat = repondre(
                    store(),
                    question,
                    texte_cp,
                    contrat_id
                )
                reponse = resultat["reponse"]
                t2 = time.perf_counter()
                reponse = resultat["reponse"]
                passages = resultat["passages"]
                infos = f"génération et recherche : {t2 - t0:.1f} s"
        st.markdown(reponse)
        st.caption(infos)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reponse,
        "infos": infos,
        "passages": [(f"{p.metadata['fichier']} p.{p.metadata['page']}", p.page_content[:600]) for p in passages],
    })
    st.rerun()