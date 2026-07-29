"""Funcții utilitare: încărcarea datelor și normalizarea textului în limba română."""

import re
import unicodedata
from pathlib import Path

import pandas as pd

# Rădăcina proiectului (directorul părinte al lui src/)
ROOT = Path(__file__).resolve().parent.parent
FAQ_PATH = ROOT / "data" / "faq.csv"
TEST_PATH = ROOT / "data" / "intrebari_test.csv"

# Listă scurtă de cuvinte de legătură (stopwords) pentru limba română.
STOPWORDS = {
    "a", "ai", "al", "ale", "am", "ar", "as", "asta", "au", "ca", "care",
    "cand", "ce", "cel", "cine", "cu", "cum", "da", "daca", "de", "din",
    "dupa", "e", "el", "ea", "este", "eu", "fac", "face", "fi", "fie",
    "il", "imi", "in", "intre", "isi", "iti", "la", "le", "lor", "lui",
    "ma", "mai", "mea", "mele", "meu", "mi", "ne", "ni", "nu", "o", "or",
    "pe", "pentru", "poate", "pot", "prin", "sa", "sau", "se", "si",
    "sunt", "te", "ti", "trebuie", "un", "una", "unde", "unei", "unui",
    "va", "vor", "vreau",
}


def strip_diacritics(text: str) -> str:
    """Elimină diacriticele (ș -> s, ă -> a etc.).

    Utilizatorii scriu adesea fără diacritice, deci normalizarea face
    căutarea robustă la ambele variante de scriere.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Aduce textul la o formă canonică: litere mici, fără diacritice,
    fără semne de punctuație, spații multiple reduse la unul singur."""
    text = strip_diacritics(str(text).lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Împarte textul normalizat în cuvinte, opțional fără stopwords."""
    tokens = normalize(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def load_faq(path: Path | str = FAQ_PATH) -> pd.DataFrame:
    """Încarcă baza de întrebări frecvente și verifică structura minimă."""
    df = pd.read_csv(path)
    expected = {"id", "categorie", "intrebare", "raspuns"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Coloane lipsă în {path}: {missing}")
    if df["intrebare"].isnull().any() or df["raspuns"].isnull().any():
        raise ValueError("Există întrebări sau răspunsuri lipsă în setul de date.")
    return df


def load_test_questions(path: Path | str = TEST_PATH) -> pd.DataFrame:
    """Încarcă setul de întrebări de test (parafraze cu răspuns așteptat)."""
    df = pd.read_csv(path)
    expected = {"id_asteptat", "intrebare_test"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Coloane lipsă în {path}: {missing}")
    return df
