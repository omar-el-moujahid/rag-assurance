"""Fournit le traçage choisi dans config.TRACING, sans que le reste du code s'en occupe."""
from config import TRACING

TYPES = {   # notre type -> (type LangSmith, type Langfuse)
    "chain":      ("chain", "chain"),
    "retriever":  ("retriever", "retriever"),
    "generation": ("llm", "generation"),
}

def tracer(nom, type_="chain"):
    """Trace la fonction avec l'outil choisi dans config.TRACING (ou ne fait rien)."""
    type_langsmith, type_langfuse = TYPES[type_]
    def decorer(fonction):
        if TRACING == "langfuse":
            from langfuse import observe
            return observe(name=nom, as_type=type_langfuse)(fonction)
        if TRACING == "langsmith":
            from langsmith import traceable
            return traceable(name=nom, run_type=type_langsmith)(fonction)
        return fonction
    return decorer

def callbacks():
    """Les callbacks à passer aux appels LangChain."""
    if TRACING == "langfuse":
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    return []          # LangSmith n'a pas besoin de callback : il lit les variables d'environnement


def envoyer_traces():
    """À appeler à la fin d'un script court, pour ne pas quitter avant l'envoi des traces."""
    if TRACING == "langfuse":
        from langfuse import get_client
        get_client().flush()