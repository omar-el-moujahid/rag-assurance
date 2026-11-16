"""Étape 3 : extraction des CP.
CP (PDF) -> texte -> fiche (LLM, sortie structurée) -> contrat_id (règles déterministes).
Test :  python -m src.extraction_cp.cp_extraction data/cp/cp_03_sogessur_F190315.pdf
"""
import json
import re
import sys
import time
import unicodedata
from typing import Optional

from pydantic import BaseModel, Field
from pypdf import PdfReader
from langchain_ollama import ChatOllama
from langsmith import traceable
from src.tracing.tracing import tracer

from config import CATALOGUE, LLM_MODEL

catalogue = json.loads(CATALOGUE.read_text(encoding="utf-8"))

MOIS = {"janvier": "01", "fevrier": "02", "mars": "03", "avril": "04", "mai": "05", "juin": "06",
        "juillet": "07", "aout": "08", "septembre": "09", "octobre": "10", "novembre": "11", "decembre": "12"}

alternative_model = "qwen3:8b"  # pour tester l'extraction si le modèle principal est indisponible


class FicheCP(BaseModel):
    assureur: str = Field(max_length=80,
        description="Nom de l'assureur tel qu'écrit dans le document")
    reference_cg: Optional[str] = Field(max_length=30,
        description="CODE alphanumérique de référence des Conditions Générales (ex : 'F 190 315', 'HCE 2504'). "
                    "Ce n'est NI le nom du produit NI une date. Mets null si le document ne donne pas de code.")
    edition_cg: Optional[str] = Field(max_length=30,
        description="Date d'édition des Conditions Générales, AVEC le mois et l'année si présents "
                    "(ex : '04/2025', 'janvier 2023'). null si absente.")
    formule: Optional[str] = Field(max_length=50,
        description="Nom court de la formule souscrite, tel qu'indiqué dans la section des garanties "
                    "(ex : 'Couverture Étendue', 'Formule Confort'). null si absente.")


CONSIGNE = """Tu extrais des informations de conditions particulières (CP) d'assurance habitation.
Règles :
- Recopie les valeurs exactement comme dans le document, sans les reformuler.
- Si une information est absente du document, mets null. N'invente jamais.

Document :
{texte}"""

# Créés une seule fois, au chargement du module
llm = ChatOllama(model=LLM_MODEL, temperature=0, num_predict=800, keep_alive=0)
extracteur = llm.with_structured_output(FicheCP, method="json_schema")


def lire_cp(chemin) -> str:
    """Texte de toutes les pages de la CP."""
    return "\n".join(page.extract_text() or "" for page in PdfReader(chemin).pages)


def extraire_fiche(texte: str) -> FicheCP:
    """Le LLM remplit la fiche à partir du texte de la CP."""
    return extracteur.invoke(CONSIGNE.format(texte=texte))


def normaliser(s) -> str:
    """'Société Générale' -> 'societegenerale' ; 'F 190 315 - 04/2025' -> 'f190315042025'."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    for nom, num in MOIS.items():
        s = s.replace(nom, num)
    return re.sub(r"[^a-z0-9]", "", s)


def lignes_cg(texte_cp: str) -> str:
    """Les lignes de la CP qui mentionnent les Conditions Générales."""
    return " ".join(l for l in texte_cp.splitlines() if "conditions générales" in l.lower())


def trouver_contrat(fiche: FicheCP, texte_cp: str) -> Optional[str]:
    """Règles déterministes : assureur, puis référence, puis édition. Un seul candidat, sinon None."""
    assureur = normaliser(fiche.assureur)
    ref_fiche = normaliser(f"{fiche.reference_cg} {lignes_cg(texte_cp)}")
    ref_et_edition = normaliser(f"{fiche.reference_cg} {fiche.edition_cg} {lignes_cg(texte_cp)}")

    candidats = set()
    for e in catalogue:
        if e["type_doc"] != "CG":
            continue
        if not any(normaliser(nom) in assureur for nom in [e["assureur"]] + e["alias"]):
            continue
        if e["reference"] and normaliser(e["reference"]) not in ref_fiche:
            continue
        if e["edition"] and normaliser(e["edition"]) not in ref_et_edition:
            continue
        candidats.add(e["contrat_id"])

    return candidats.pop() if len(candidats) == 1 else None

@tracer("analyser_cp", "chain")
def analyser_cp(chemin):
    texte = lire_cp(chemin)
    try:
        fiche = extraire_fiche(texte)
    except Exception as e:                       # sortie vide, JSON invalide, Ollama injoignable...
        print(f"⚠️ Extraction impossible pour {chemin} : {type(e).__name__}")
        return texte, None, None
    return texte, fiche, trouver_contrat(fiche, texte)


if __name__ == "__main__":
    debut = time.time()
    texte, fiche, contrat_id = analyser_cp(sys.argv[1])
    if fiche is not None:
        print(fiche.model_dump_json(indent=2))
    print(f"contrat_id = {contrat_id}")
    print(f"Temps d'extraction : {time.time() - debut:.1f} s")