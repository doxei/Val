"""Connaissances de Valdar (le « RAG ») : ce qu'il sait sur l'impression 3D et ce qu'il apprend.

100 % local, sans modèle : BM25 sur les mots normalisés + n-grammes hachés (mêmes vecteurs que
la mémoire) pour tolérer les fautes de la dictée. Les sources : la base de défauts d'impression
(docs/connaissances), l'état de l'art de la détection, les notes de PrintOS, et tout document
qu'Olivier dépose dans data/connaissances/ (texte, markdown, yaml).
"""
from valdar.knowledge.store import Knowledge, Passage

__all__ = ["Knowledge", "Passage"]
