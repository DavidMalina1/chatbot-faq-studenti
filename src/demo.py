"""Demonstrație în linia de comandă (fără interfață web).

Rulare:
    python src/demo.py                       # mod interactiv
    python src/demo.py "cum iau bursa"       # o singură întrebare
    python src/demo.py --metoda ensemble "unde vad orarul"
"""

import argparse

from search import build_engines
from utils import load_faq


def afiseaza(rezultate) -> None:
    for loc, (_, r) in enumerate(rezultate.iterrows(), start=1):
        print(f"\n[{loc}] (scor {r['scor']:.2f}, categoria: {r['categorie']})")
        print(f"    Î: {r['intrebare']}")
        print(f"    R: {r['raspuns']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intrebare", nargs="*", help="întrebarea adresată")
    parser.add_argument("--metoda", default="lsa",
                        choices=["keyword", "tfidf", "lsa", "ensemble", "sbert"])
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    faq = load_faq()
    motoare = build_engines(faq, include_sbert=(args.metoda == "sbert"))
    motor = motoare[args.metoda]
    print(f"Chatbot FAQ – metoda: {args.metoda}, "
          f"{len(faq)} întrebări în baza de cunoștințe.")

    if args.intrebare:
        afiseaza(motor.search(" ".join(args.intrebare), top_k=args.top))
        return

    print("Scrie o întrebare (sau 'exit' pentru a ieși).")
    while True:
        try:
            q = input("\nÎntrebare > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit", "iesire"}:
            break
        afiseaza(motor.search(q, top_k=args.top))


if __name__ == "__main__":
    main()
