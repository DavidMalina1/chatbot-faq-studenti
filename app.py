"""Interfața web a chatbotului FAQ pentru studenți (Streamlit).

Rulare:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from search import build_engines  # noqa: E402
from utils import ROOT, load_faq  # noqa: E402

PRAG_INCREDERE = {"keyword": 0.12, "tfidf": 0.28, "lsa": 0.35,
                  "ensemble": 0.30, "sbert": 0.45}
ETICHETE = {
    "keyword": "1. Cuvinte-cheie (varianta de bază)",
    "tfidf": "2. TF-IDF + similaritate cosinus",
    "lsa": "3. Căutare semantică (LSA)",
    "ensemble": "4. Combinat (medie ponderată)",
    "sbert": "SBERT (opțional)",
}
CULORI_CATEGORII = {
    "secretariat": "#4a7fb5", "examene": "#b85042", "burse": "#2c5f2d",
    "cazare": "#7b4fa6", "taxe": "#c07f2a", "studii": "#1e2761",
    "practica": "#0e7c86", "licenta": "#8a2f62", "elearning": "#4f5d75",
    "erasmus": "#4e6e35",
}
EXEMPLE = [
    "Ce dosar trebuie sa fac pentru bursa sociala?",
    "Daca am medie mare pot ajunge la buget?",
    "Ma pot muta la alta specializare?",
    "Cand imi aleg subiectul pentru licenta?",
]
REZULTATE_DIR = ROOT / "reports" / "rezultate"

st.set_page_config(page_title="Chatbot FAQ Studenți", page_icon="🎓",
                   layout="centered")


@st.cache_resource
def incarca_motoare(include_sbert: bool):
    faq = load_faq()
    return faq, build_engines(faq, include_sbert=include_sbert)


def badge(categorie: str) -> str:
    culoare = CULORI_CATEGORII.get(categorie, "#666666")
    return (f"<span style='background:{culoare};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.75rem'>{categorie}</span>")


def raspunde(intrebare: str, motor, metoda: str, top_k: int):
    """Construiește răspunsul (markdown) pentru o întrebare."""
    rezultate = motor.search(intrebare, top_k=top_k)
    prag = PRAG_INCREDERE.get(metoda, 0.3)
    top = rezultate.iloc[0]
    if top["scor"] < prag:
        return ("Nu am găsit un răspuns suficient de potrivit în baza de "
                "întrebări frecvente. 🤔 Încearcă să reformulezi întrebarea "
                "sau contactează secretariatul facultății."), rezultate, False
    text = (f"{badge(top['categorie'])}\n\n**{top['intrebare']}**\n\n"
            f"{top['raspuns']}")
    return text, rezultate, True


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Setări")
    include_sbert = st.checkbox(
        "Activează SBERT (necesită modelul descărcat)", value=False)
    faq, motoare = incarca_motoare(include_sbert)
    metoda = st.radio("Metoda de căutare", list(motoare.keys()),
                      format_func=lambda m: ETICHETE.get(m, m),
                      index=list(motoare.keys()).index("ensemble"))
    top_k = st.slider("Număr de răspunsuri afișate", 1, 5, 3)
    st.divider()
    st.markdown(f"📚 **{len(faq)}** întrebări · "
                f"**{faq['categorie'].nunique()}** categorii")
    nr = len([m for m in st.session_state.get("istoric", [])
              if m["rol"] == "user"])
    st.markdown(f"💬 Întrebări în această sesiune: **{nr}**")
    with st.expander("Categoriile disponibile"):
        st.markdown(" ".join(badge(c) for c in
                             sorted(faq["categorie"].unique())),
                    unsafe_allow_html=True)

# ----------------------------------------------------------------- taburi
tab_chat, tab_stats = st.tabs(["💬 Chat", "📊 Statistici"])

with tab_chat:
    st.title("🎓 Chatbot FAQ pentru studenți")
    st.caption("Întreabă orice despre secretariat, examene, burse, cazare, "
               "taxe, practică, licență, e-learning sau Erasmus.")

    # butoane cu întrebări-exemplu
    st.markdown("**Încearcă un exemplu:**")
    coloane = st.columns(2)
    intrebare_exemplu = None
    for i, ex in enumerate(EXEMPLE):
        if coloane[i % 2].button(ex, use_container_width=True, key=f"ex{i}"):
            intrebare_exemplu = ex

    if "istoric" not in st.session_state:
        st.session_state.istoric = []

    for mesaj in st.session_state.istoric:
        with st.chat_message(mesaj["rol"]):
            st.markdown(mesaj["continut"], unsafe_allow_html=True)
            if mesaj.get("alternative"):
                with st.expander("Alte răspunsuri posibile"):
                    for alt in mesaj["alternative"]:
                        st.markdown(alt, unsafe_allow_html=True)

    intrebare = st.chat_input("Scrie întrebarea ta aici...")
    if intrebare_exemplu and not intrebare:
        intrebare = intrebare_exemplu

    if intrebare:
        st.session_state.istoric.append({"rol": "user",
                                         "continut": intrebare})
        with st.chat_message("user"):
            st.markdown(intrebare)

        text, rezultate, gasit = raspunde(intrebare, motoare[metoda],
                                          metoda, top_k)
        alternative = []
        if gasit and len(rezultate) > 1:
            for _, r in rezultate.iloc[1:].iterrows():
                alternative.append(
                    f"{badge(r['categorie'])} **{r['intrebare']}** — "
                    f"scor {r['scor']:.2f}")

        with st.chat_message("assistant"):
            st.markdown(text, unsafe_allow_html=True)
            if gasit:
                scor = float(rezultate.iloc[0]["scor"])
                st.progress(min(scor, 1.0),
                            text=f"Încredere: {scor:.2f} · metoda: "
                                 f"{ETICHETE[metoda]}")
            if alternative:
                with st.expander("Alte răspunsuri posibile"):
                    for alt in alternative:
                        st.markdown(alt, unsafe_allow_html=True)

        st.session_state.istoric.append({
            "rol": "assistant", "continut": text,
            "alternative": alternative,
        })

with tab_stats:
    st.subheader("📊 Rezultatele evaluării comparative")
    st.caption("Generate cu `python src/evaluate.py` pe cele 40 de "
               "întrebări de test.")
    metrici_path = REZULTATE_DIR / "metrici.csv"
    if metrici_path.exists():
        metrici = pd.read_csv(metrici_path)
        metrici["metoda"] = metrici["metoda"].map(
            lambda m: ETICHETE.get(m, m))
        st.dataframe(metrici, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        img1 = REZULTATE_DIR / "recall_comparatie.png"
        img2 = REZULTATE_DIR / "recall_cumulat.png"
        if img1.exists():
            col1.image(str(img1), caption="recall@K pe metode")
        if img2.exists():
            col2.image(str(img2), caption="recall cumulat în funcție de K")
        st.markdown(
            "**Interpretare pe scurt:** varianta de bază (cuvinte-cheie) "
            "rezolvă din prima jumătate din întrebări; TF-IDF și căutarea "
            "semantică urcă semnificativ, iar **metoda combinată** — media "
            "ponderată a scorurilor celor trei — obține cele mai bune "
            "rezultate: metodele greșesc pe întrebări diferite și se "
            "corectează reciproc.")
    else:
        st.info("Rezultatele nu au fost încă generate. Rulează în terminal: "
                "`python src/evaluate.py`")
