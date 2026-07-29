"""Motoarele de căutare ale chatbotului.

Sunt implementate trei variante, de la simplu la complex:

1. KeywordSearch  - varianta de bază: suprapunere de cuvinte-cheie (Jaccard);
2. TfidfSearch    - vectorizare TF-IDF (cuvinte + n-grame de caractere)
                    și similaritate cosinus;
3. LsaSearch      - căutare semantică latentă (LSA): TF-IDF proiectat
                    într-un spațiu de dimensiune redusă cu TruncatedSVD,
                    apoi similaritate cosinus între embeddings.

Opțional (necesită internet la prima rulare pentru descărcarea modelului):

4. SbertSearch    - embeddings neuronale multilingve cu sentence-transformers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize as l2_normalize

from utils import normalize, tokenize


class BaseSearch:
    """Interfață comună: primește DataFrame-ul FAQ, oferă metoda search()."""

    name = "base"

    def __init__(self, faq: pd.DataFrame):
        self.faq = faq.reset_index(drop=True)

    def scores(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """Returnează primele top_k răspunsuri, ordonate după scor."""
        s = self.scores(query)
        order = np.argsort(-s)[:top_k]
        result = self.faq.iloc[order].copy()
        result["scor"] = s[order]
        return result


class KeywordSearch(BaseSearch):
    """Varianta de bază: similaritate Jaccard pe mulțimi de cuvinte-cheie.

    scor(q, d) = |cuvinte(q) ∩ cuvinte(d)| / |cuvinte(q) ∪ cuvinte(d)|
    """

    name = "keyword"

    def __init__(self, faq: pd.DataFrame):
        super().__init__(faq)
        self._doc_tokens = [set(tokenize(q)) for q in self.faq["intrebare"]]

    def scores(self, query: str) -> np.ndarray:
        q = set(tokenize(query))
        out = np.zeros(len(self._doc_tokens))
        if not q:
            return out
        for i, d in enumerate(self._doc_tokens):
            union = q | d
            if union:
                out[i] = len(q & d) / len(union)
        return out


def _build_vectorizer() -> FeatureUnion:
    """TF-IDF combinat: cuvinte (1-2 grame) + n-grame de caractere (3-5).

    N-gramele de caractere fac potrivirea robustă la flexiuni
    (restanta/restantele), la diacritice și la mici greșeli de tastare.
    """
    return FeatureUnion([
        ("cuvinte", TfidfVectorizer(preprocessor=normalize,
                                    ngram_range=(1, 2), sublinear_tf=True)),
        ("caractere", TfidfVectorizer(preprocessor=normalize,
                                      analyzer="char_wb",
                                      ngram_range=(3, 5), sublinear_tf=True)),
    ])


class TfidfSearch(BaseSearch):
    """Îmbunătățirea 1: vectori TF-IDF + similaritate cosinus."""

    name = "tfidf"

    def __init__(self, faq: pd.DataFrame):
        super().__init__(faq)
        self.vectorizer = _build_vectorizer()
        docs = self.faq["intrebare"].tolist()
        self._doc_matrix = l2_normalize(self.vectorizer.fit_transform(docs))

    def scores(self, query: str) -> np.ndarray:
        q_vec = l2_normalize(self.vectorizer.transform([query]))
        # Similaritatea cosinus = produs scalar între vectori normalizați L2.
        return np.asarray((self._doc_matrix @ q_vec.T).todense()).ravel()


class LsaSearch(BaseSearch):
    """Îmbunătățirea 2: căutare semantică latentă (LSA).

    Reprezentarea TF-IDF este proiectată cu TruncatedSVD într-un spațiu
    dens de dimensiune redusă (embeddings). Dimensiunile latente grupează
    termeni care apar în contexte similare, astfel încât o interogare poate
    fi apropiată de o întrebare din FAQ chiar fără cuvinte identice.
    """

    name = "lsa"

    def __init__(self, faq: pd.DataFrame, n_components: int = 30,
                 random_state: int = 42):
        super().__init__(faq)
        self.vectorizer = _build_vectorizer()
        # Documentele indexate combină întrebarea (pondere dublă) și
        # răspunsul: răspunsul aduce vocabular suplimentar util pentru SVD.
        docs = (self.faq["intrebare"] + " " + self.faq["intrebare"] + " "
                + self.faq["raspuns"]).tolist()
        tfidf = self.vectorizer.fit_transform(docs)
        n_components = min(n_components, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components,
                                random_state=random_state)
        self._doc_emb = l2_normalize(self.svd.fit_transform(tfidf))

    def embed(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self.svd.transform(self.vectorizer.transform(texts)))

    def scores(self, query: str) -> np.ndarray:
        q_emb = self.embed([query])
        return (self._doc_emb @ q_emb.T).ravel()


class SbertSearch(BaseSearch):
    """Extensie opțională: embeddings neuronale cu sentence-transformers.

    Folosește un model multilingv preantrenat. La prima rulare, modelul
    (~470 MB) este descărcat automat de pe Hugging Face, deci este
    necesară o conexiune la internet.
    """

    name = "sbert"
    MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, faq: pd.DataFrame, model_name: str | None = None):
        super().__init__(faq)
        from sentence_transformers import SentenceTransformer  # import leneș
        self.model = SentenceTransformer(model_name or self.MODEL)
        self._doc_emb = self.model.encode(self.faq["intrebare"].tolist(),
                                          normalize_embeddings=True)

    def scores(self, query: str) -> np.ndarray:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        return (self._doc_emb @ q_emb.T).ravel()


def build_engines(faq: pd.DataFrame, include_sbert: bool = False) -> dict:
    """Construiește toate motoarele de căutare disponibile."""
    engines: dict[str, BaseSearch] = {
        "keyword": KeywordSearch(faq),
        "tfidf": TfidfSearch(faq),
        "lsa": LsaSearch(faq),
    }
    if include_sbert:
        try:
            engines["sbert"] = SbertSearch(faq)
        except Exception as exc:  # bibliotecă lipsă sau model nedescărcat
            print(f"[avertisment] SbertSearch indisponibil: {exc}")
    return engines
