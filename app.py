"""Interfața web a chatbotului FAQ pentru studenți (Streamlit).

Rulare:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from search import build_engines  # noqa: E402
from utils import load_faq  # noqa: E402

PRAG_INCREDERE = {"keyword": 0.12, "tfidf": 0.28, "lsa": 0.35, "sbert": 0.45}
ETICHETE = {
    "keyword": "Cuvinte-cheie (varianta de bază)",
    "tfidf": "TF-IDF + similaritate cosinus",
    "lsa": "Căutare semantică (LSA)",
    "sbert": "Căutare semantică (SBERT, opțional)",
}

st.set_page_config(page_title="Chatbot FAQ Studenți", page_icon="🎓")


@st.cache_resource
def incarca_motoare(include_sbert: bool):
    faq = load_faq()
    return faq, build_engines(faq, include_sbert=include_sbert)


st.title("🎓 Chatbot FAQ pentru studenți")
st.caption("Răspunde întrebărilor frecvente despre secretariat, examene, "
           "burse, cazare, practică, licență și altele.")

with st.sidebar:
    st.header("Setări")
    include_sbert = st.checkbox(
        "Activează SBERT (necesită modelul descărcat)", value=False)
    faq, motoare = incarca_motoare(include_sbert)
    metoda = st.radio("Metoda de căutare",
                      list(motoare.keys()),
                      format_func=lambda m: ETICHETE.get(m, m),
                      index=min(2, len(motoare) - 1))
    top_k = st.slider("Număr de răspunsuri afișate", 1, 5, 3)
    st.divider()
    st.markdown(f"**{len(faq)}** întrebări în baza de cunoștințe, "
                f"**{faq['categorie'].nunique()}** categorii.")
    with st.expander("Vezi toate categoriile"):
        st.write(", ".join(sorted(faq["categorie"].unique())))

if "istoric" not in st.session_state:
    st.session_state.istoric = []

for mesaj in st.session_state.istoric:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["continut"])

intrebare = st.chat_input("Scrie întrebarea ta aici...")

if intrebare:
    st.session_state.istoric.append({"rol": "user", "continut": intrebare})
    with st.chat_message("user"):
        st.markdown(intrebare)

    rezultate = motoare[metoda].search(intrebare, top_k=top_k)
    prag = PRAG_INCREDERE.get(metoda, 0.3)
    cel_mai_bun = rezultate.iloc[0]

    with st.chat_message("assistant"):
        if cel_mai_bun["scor"] < prag:
            raspuns = ("Nu am găsit un răspuns suficient de potrivit în baza "
                       "de întrebări frecvente. Încearcă să reformulezi "
                       "întrebarea sau contactează secretariatul facultății.")
            st.markdown(raspuns)
        else:
            raspuns = (f"**{cel_mai_bun['intrebare']}**\n\n"
                       f"{cel_mai_bun['raspuns']}\n\n"
                       f"*Categorie: {cel_mai_bun['categorie']} · "
                       f"scor: {cel_mai_bun['scor']:.2f}*")
            st.markdown(raspuns)
            if len(rezultate) > 1:
                with st.expander("Alte răspunsuri posibile"):
                    for _, r in rezultate.iloc[1:].iterrows():
                        st.markdown(f"- **{r['intrebare']}** "
                                    f"(scor {r['scor']:.2f})")
        st.session_state.istoric.append({"rol": "assistant",
                                         "continut": raspuns})
