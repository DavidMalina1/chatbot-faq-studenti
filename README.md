# Chatbot Q&A pentru întrebări frecvente ale studenților

Proiect de practică (Tema 3 din Ghidul de practică CTI 2026): un sistem care
răspunde întrebărilor studenților pe baza unei colecții de întrebări și
răspunsuri frecvente (secretariat, examene, burse, cazare, taxe, practică,
licență, e-learning, Erasmus).

## Descrierea problemei

Studenții adresează frecvent aceleași întrebări administrative, formulate însă
foarte diferit („Cum vorbesc cu tutorele de an?" vs. „Cum pot lua legătura cu
îndrumătorul de an?"). Sistemul primește o întrebare în limbaj natural și
returnează cel mai potrivit răspuns din baza de cunoștințe, comparând trei
metode de căutare, de la simplă la semantică.

## Tehnologii utilizate

- **Python 3.10+**, **pandas**, **NumPy** – încărcarea și prelucrarea datelor;
- **scikit-learn** – TF-IDF, TruncatedSVD (LSA), normalizare;
- **Matplotlib** – graficele de evaluare;
- **Streamlit** – interfața web de tip chat;
- opțional: **sentence-transformers** – embeddings neuronale multilingve.

## Structura proiectului

```
chatbot-faq-studenti/
├── data/
│   ├── faq.csv                # baza de cunoștințe: 40 întrebări + răspunsuri, 10 categorii
│   └── intrebari_test.csv     # 40 de parafraze de test cu răspunsul așteptat
├── src/
│   ├── utils.py               # încărcarea datelor, normalizare text (diacritice, stopwords)
│   ├── search.py              # cele 3(+1) motoare de căutare
│   ├── evaluate.py            # evaluare recall@K, MRR, grafice
│   └── demo.py                # demonstrație în linia de comandă
├── reports/
│   └── rezultate/             # metrici.csv, detalii_*.csv, grafice PNG
├── app.py                     # aplicația web Streamlit
├── requirements.txt
└── README.md
```

## Metodele implementate

1. **Cuvinte-cheie (varianta de bază)** – similaritate Jaccard între mulțimile
   de cuvinte ale întrebării utilizatorului și ale întrebărilor din FAQ, după
   normalizare (litere mici, fără diacritice, fără stopwords).
2. **TF-IDF + similaritate cosinus** – vectorizare combinată pe cuvinte
   (1–2 grame) și n-grame de caractere (3–5), robustă la flexiuni și greșeli
   de tastare.
3. **Căutare semantică (LSA)** – embeddings obținute prin proiectarea
   reprezentării TF-IDF într-un spațiu latent de dimensiune redusă
   (TruncatedSVD), apoi similaritate cosinus.
4. *(opțional)* **SBERT** – embeddings neuronale cu modelul multilingv
   `paraphrase-multilingual-MiniLM-L12-v2` (necesită internet la prima rulare
   pentru descărcarea modelului, ~470 MB).

## Instalare

```bash
# 1. Clonați / dezarhivați proiectul, apoi, din directorul proiectului:
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 2. Instalați dependențele
pip install -r requirements.txt

# (opțional, pentru varianta SBERT)
pip install sentence-transformers
```

## Rulare

**Aplicația web (recomandat pentru demonstrație):**

```bash
streamlit run app.py
```

Aplicația se deschide în browser la `http://localhost:8501`. Din bara laterală
se pot alege metoda de căutare și numărul de răspunsuri afișate.

**Demonstrație în linia de comandă:**

```bash
python src/demo.py "cum pot lua bursa de merit"
python src/demo.py --metoda keyword "unde vad orarul"
python src/demo.py            # mod interactiv
```

**Evaluarea comparativă (regenerează tabelele și graficele):**

```bash
python src/evaluate.py            # keyword + tfidf + lsa
python src/evaluate.py --sbert    # include și SBERT (dacă e instalat)
```

## Sursa datelor

Ambele seturi de date au fost **create manual** pentru acest proiect:

- `data/faq.csv` – 40 de perechi întrebare–răspuns despre viața de student,
  inspirate din regulamentele și paginile publice tipice ale facultăților
  tehnice din România (conținutul este generic, nu reproduce documentele unei
  instituții anume);
- `data/intrebari_test.csv` – 40 de reformulări realiste ale întrebărilor
  (scrise fără diacritice, cu sinonime și structuri diferite), folosite
  exclusiv pentru evaluare, nu și la construirea indexului.

## Principalele rezultate

Evaluare pe cele 40 de întrebări de test (răspunsul corect trebuie să apară
în primele K rezultate):

| Metodă | recall@1 | recall@3 | recall@5 | MRR | timp mediu / întrebare |
|---|---|---|---|---|---|
| Cuvinte-cheie (bază) | 0.50 | 0.68 | 0.80 | 0.62 | ~1 ms |
| TF-IDF + cosinus | 0.78 | **0.95** | 0.98 | 0.86 | ~4 ms |
| Semantic (LSA) | **0.83** | 0.93 | **0.98** | **0.88** | ~4 ms |

Căutarea semantică rezolvă corect din prima încercare 33/40 de întrebări,
față de 20/40 pentru varianta de bază. Detaliile per întrebare și graficele
se află în `reports/rezultate/`, iar interpretarea completă și analiza
erorilor în raportul scris.

## Exemple de utilizare

```
Întrebare > cum pot lua bursa de merit
[1] (scor 0.79, categoria: burse)
    Î: Care este media minimă pentru bursa de merit?
    R: Media minimă pentru bursa de merit variază în funcție de fondul de
       burse... este necesară o medie de cel puțin 8.50...
```

Dacă scorul celui mai bun rezultat este sub pragul de încredere, aplicația
recunoaște că nu știe răspunsul și recomandă contactarea secretariatului.

## Limitări

- Baza de cunoștințe este mică (40 de intrări) și cu conținut generic;
- LSA rămâne o metodă lexical-latentă: sinonimele complet diferite
  („mail" vs. „e-mail instituțional") pot fi ratate — varianta SBERT
  adresează exact acest caz;
- Sistemul returnează răspunsuri existente, nu generează texte noi.
