"""
Confronto didattico: BFGS e MultiStart scritti a mano contro SciPy.

Esegui dalla cartella del progetto:

    python test/test.py

Non modifica gli algoritmi. Li chiama, stampa come si comportano
su funzioni di cui il minimo è noto, e salva i grafici in figure/.

Cosa guardare in ogni riga:
- f            valore della funzione nel punto trovato (più basso è meglio)
- ||x - x*||   distanza dal minimo noto
- ||grad||     norma euclidea del gradiente: vicina a zero solo se è un punto stazionario
- valutazioni  quante volte è stata chiamata f

Differenze volute, da tenere presenti mentre leggi i numeri:
- il BFGS a mano fa sempre 101 passi (il contatore interno va da 0 a 100)
  e non si ferma quando il gradiente è già piccolo;
- il line search a mano è Armijo (solo diminuzione di f). SciPy BFGS usa Wolfe;
- SciPy BFGS si ferma se la norma infinito del gradiente scende sotto 1e-5;
- MultiStart pesca ogni coordinata con random.randint(1, 10).

Su Rastrigin questo ultimo punto cambia la lettura del risultato: in un intero
il gradiente è 2x, il primo passo con B = I e α = 1/2 cade sull'origine, e
Armijo lo accetta. Il multistart sembra "globale" perché ogni lancio parte da
un intero, non perché confronti dei bacini.
"""

import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, minimize

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    plt = None
    Line2D = None

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from BFGS import BFGS
from multistart import MultiStart

SEED = 0
N_PARTENZE = 8
# Norma infinito del gradiente sotto la quale SciPy BFGS dichiara successo.
GTOL_SCIPY = 1e-5


# ---------------------------------------------------------------------------
# Funzioni test. Accanto a ognuna c'è il suo gradiente analitico.
# ---------------------------------------------------------------------------

TARGET_QUADRATICA = np.array([1.0, -2.0])
H_QUADRATICA = np.array([[100.0, 0.0], [0.0, 0.05]])


def quadratica(x):
    """Quadratica convessa mal condizionata. Minimo esatto in (1, -2)."""
    d = np.asarray(x, dtype=float) - TARGET_QUADRATICA
    return float(0.5 * d @ H_QUADRATICA @ d)


def nabla_quadratica(x):
    d = np.asarray(x, dtype=float) - TARGET_QUADRATICA
    return H_QUADRATICA @ d


def rosenbrock(x):
    """Valle stretta e curva. Minimo esatto in (1, 1), f* = 0."""
    x = np.asarray(x, dtype=float)
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def nabla_rosenbrock(x):
    x = np.asarray(x, dtype=float)
    a, b = x[0], x[1]
    return np.array(
        [-2.0 * (1.0 - a) - 400.0 * a * (b - a ** 2), 200.0 * (b - a ** 2)]
    )


# Tre buche gaussiane. Quella centrata in (7, 3) è la più profonda.
# Sigma largo: con buche troppo strette il gradiente è circa zero quasi ovunque
# e sia il BFGS sia i metodi globali si fermano sul pianoro, non in una buca.
SIGMA_POZZI = 1.5
POZZI = (
    (np.array([2.0, 8.0]), 3.0),
    (np.array([7.0, 3.0]), 8.0),
    (np.array([4.0, 2.0]), 1.5),
)


def multi_pozzo(x):
    x = np.asarray(x, dtype=float)
    varianza = SIGMA_POZZI ** 2
    valore = 0.0
    for centro, profondita in POZZI:
        r = float(np.sum((x - centro) ** 2)) / (2.0 * varianza)
        valore -= profondita * math.exp(-r)
    return valore


def nabla_multi_pozzo(x):
    x = np.asarray(x, dtype=float)
    varianza = SIGMA_POZZI ** 2
    gradiente = np.zeros_like(x)
    for centro, profondita in POZZI:
        scarto = x - centro
        r = float(np.sum(scarto ** 2)) / (2.0 * varianza)
        gradiente += (profondita / varianza) * math.exp(-r) * scarto
    return gradiente


def himmelblau(x):
    """Quattro minimi globali, tutti con f* = 0. Uno solo, (3, 2), sta in [1, 10]^2."""
    x = np.asarray(x, dtype=float)
    a, b = x[0], x[1]
    return float((a ** 2 + b - 11.0) ** 2 + (a + b ** 2 - 7.0) ** 2)


def nabla_himmelblau(x):
    x = np.asarray(x, dtype=float)
    a, b = x[0], x[1]
    p = a ** 2 + b - 11.0
    q = a + b ** 2 - 7.0
    return np.array([4.0 * a * p + 2.0 * q, 2.0 * p + 4.0 * b * q])


def rastrigin(x):
    """Tanti minimi locali vicini agli interi. L'unico globale è l'origine, f* = 0."""
    x = np.asarray(x, dtype=float)
    return float(10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2.0 * math.pi * x)))


def nabla_rastrigin(x):
    x = np.asarray(x, dtype=float)
    return 2.0 * x + 20.0 * math.pi * np.sin(2.0 * math.pi * x)


# ---------------------------------------------------------------------------
# Utilità di stampa e di misura
# ---------------------------------------------------------------------------

def fmt_x(x, cifre=5):
    valori = ", ".join(f"{v: .{cifre}f}" for v in np.asarray(x, dtype=float))
    return "(" + valori + ")"


def fmt_num(valore):
    if valore is None or not np.isfinite(valore):
        return "      n/d"
    assoluto = abs(valore)
    if assoluto != 0.0 and (assoluto < 1e-3 or assoluto >= 1e4):
        return f"{valore: .2e}"
    return f"{valore: .6f}"


def fmt_tempo(secondi):
    if secondi < 1.0:
        return f"{secondi * 1000:.1f} ms"
    return f"{secondi:.2f} s"


def fmt_bounds(bounds):
    return " × ".join(f"[{a:g}, {b:g}]" for a, b in bounds)


def separatore(titolo):
    print()
    print("=" * 78)
    print(titolo)
    print("=" * 78)


def distanza(x, riferimento):
    x = np.asarray(x, dtype=float)
    if isinstance(riferimento, (list, tuple)):
        return min(float(np.linalg.norm(x - np.asarray(p))) for p in riferimento)
    return float(np.linalg.norm(x - np.asarray(riferimento, dtype=float)))


def norma_gradiente(nabla, x):
    return float(np.linalg.norm(np.asarray(nabla(x), dtype=float)))


def errore_gradiente_fd(f, nabla, x, eps=1e-6):
    """Confronta il gradiente analitico con le differenze finite centrate."""
    x = np.asarray(x, dtype=float)
    analitico = np.asarray(nabla(x), dtype=float)
    differenze = np.zeros_like(analitico)
    for i in range(len(x)):
        avanti = x.copy()
        indietro = x.copy()
        avanti[i] += eps
        indietro[i] -= eps
        differenze[i] = (f(avanti) - f(indietro)) / (2.0 * eps)
    return float(np.linalg.norm(analitico - differenze))


def controlla_gradienti():
    separatore("1. Controllo dei gradienti (differenze finite)")
    print("Se un gradiente fosse sbagliato, il confronto con SciPy non vorrebbe dire nulla.")
    prove = (
        ("quadratica mal condizionata", quadratica, nabla_quadratica, np.array([0.0, 4.0])),
        ("Rosenbrock", rosenbrock, nabla_rosenbrock, np.array([-1.2, 1.0])),
        ("multi-pozzo", multi_pozzo, nabla_multi_pozzo, np.array([5.0, 5.0])),
        ("Himmelblau", himmelblau, nabla_himmelblau, np.array([1.0, 1.0])),
        ("Rastrigin", rastrigin, nabla_rastrigin, np.array([0.3, -1.2])),
    )
    for nome, f, nabla, x in prove:
        errore = errore_gradiente_fd(f, nabla, x)
        scala = max(1.0, norma_gradiente(nabla, x))
        esito = "ok" if errore / scala < 1e-5 else "DA RIVEDERE"
        print(f"  {nome:<32} ||grad - differenze finite|| = {errore:.2e}  [{esito}]")


class Contatore:
    """Conta le chiamate senza cambiare il valore restituito."""

    def __init__(self, funzione):
        self.funzione = funzione
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return self.funzione(x)

    def azzera(self):
        self.n = 0


def esegui_bfgs_mano(x0, f, nabla):
    f_conta = Contatore(f)
    nabla_conta = Contatore(nabla)
    inizio = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            algoritmo = BFGS(np.asarray(x0, dtype=float), f_conta, nabla_conta)
            algoritmo.run()
            x = np.asarray(algoritmo.x_min, dtype=float).copy()
        if not np.all(np.isfinite(x)):
            raise ValueError("il punto trovato non è finito")
        valore = float(f(x))
        if not np.isfinite(valore):
            raise ValueError("f(x) non è finita")
        gradiente = norma_gradiente(nabla, x)
    except Exception as exc:
        return {
            "ok": False,
            "errore": f"{type(exc).__name__}: {exc}",
            "tempo": time.perf_counter() - inizio,
            "n_f": f_conta.n,
            "n_g": nabla_conta.n,
        }
    return {
        "ok": True,
        "x": x,
        "f": valore,
        "grad": gradiente,
        "tempo": time.perf_counter() - inizio,
        "n_f": f_conta.n,
        "n_g": nabla_conta.n,
    }


def esegui_bfgs_scipy(x0, f, nabla):
    inizio = time.perf_counter()
    risultato = minimize(
        f,
        np.asarray(x0, dtype=float),
        jac=nabla,
        method="BFGS",
        options={"gtol": GTOL_SCIPY},
    )
    x = np.asarray(risultato.x, dtype=float)
    return {
        "ok": bool(risultato.success) and np.all(np.isfinite(x)),
        "x": x,
        "f": float(risultato.fun),
        "grad": norma_gradiente(nabla, x) if np.all(np.isfinite(x)) else float("nan"),
        "tempo": time.perf_counter() - inizio,
        "n_f": int(risultato.nfev),
        "n_g": int(risultato.njev),
        "iter": int(risultato.nit),
        "messaggio": str(risultato.message),
    }


def stampa_riga_bfgs(etichetta, esito, riferimento, f_star):
    print(f"  {etichetta}")
    if not esito["ok"] and "x" not in esito:
        print(f"    fallito dopo {fmt_tempo(esito['tempo'])}: {esito['errore']}")
        print(f"    valutazioni di f: {esito['n_f']}, di gradiente: {esito['n_g']}")
        return
    print(
        f"    x = {fmt_x(esito['x'])}    f = {fmt_num(esito['f'])}"
        f"    f - f* = {fmt_num(esito['f'] - f_star)}"
    )
    print(
        f"    ||x - x*|| = {fmt_num(distanza(esito['x'], riferimento))}"
        f"    ||grad|| = {fmt_num(esito['grad'])}"
    )
    pezzi = [
        f"tempo {fmt_tempo(esito['tempo'])}",
        f"valutazioni f {esito['n_f']}",
        f"valutazioni gradiente {esito['n_g']}",
    ]
    if "iter" in esito:
        pezzi.append(f"iterazioni {esito['iter']}")
        pezzi.append("successo" if esito["ok"] else f"stop: {esito['messaggio']}")
    else:
        pezzi.append("iterazioni fisse 101")
    print("    " + "  |  ".join(pezzi))


def confronta_bfgs(nome, descrizione, f, nabla, riferimento, f_star, partenze):
    separatore(f"BFGS — {nome}")
    print(descrizione)
    print(f"minimo noto: {fmt_x(riferimento) if not isinstance(riferimento, list) else 'vedi sotto'}"
          f"    f* = {fmt_num(f_star)}")
    for x0 in partenze:
        print()
        print(f"partenza {fmt_x(x0, cifre=2)}")
        mano = esegui_bfgs_mano(x0, f, nabla)
        scipy = esegui_bfgs_scipy(x0, f, nabla)
        stampa_riga_bfgs("BFGS a mano", mano, riferimento, f_star)
        stampa_riga_bfgs("SciPy BFGS  ", scipy, riferimento, f_star)
        if mano["ok"] and scipy["ok"]:
            scarto = float(np.linalg.norm(mano["x"] - scipy["x"]))
            print(f"  distanza tra i due punti trovati: {fmt_num(scarto)}")


def partenze_come_multistart(n_punti, n_dim, seed):
    """Ripete la sequenza di random.randint(1, 10) usata da MultiStart."""
    random.seed(seed)
    partenze = []
    for _ in range(n_punti):
        x0 = np.zeros(n_dim)
        for i in range(n_dim):
            x0[i] = random.randint(1, 10)
        partenze.append(x0.copy())
    return partenze


def esegui_multistart_mano(f, nabla, n_punti, n_dim, seed):
    random.seed(seed)
    algoritmo = MultiStart(f, nabla, n_punti, BFGS, n_dim)
    inizio = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            algoritmo.run()
        x = None if algoritmo.x_min is None else np.asarray(algoritmo.x_min, dtype=float).copy()
        if x is None or not np.all(np.isfinite(x)):
            raise ValueError("MultiStart non ha restituito un punto finito")
        valore = float(f(x))
    except Exception as exc:
        return {
            "ok": False,
            "errore": f"{type(exc).__name__}: {exc}",
            "tempo": time.perf_counter() - inizio,
        }
    return {
        "ok": True,
        "x": x,
        "f": valore,
        "grad": norma_gradiente(nabla, x),
        "tempo": time.perf_counter() - inizio,
    }


def esegui_multistart_scipy(f, nabla, partenze):
    """Stessa idea di MultiStart: un BFGS per ogni partenza, si tiene il valore più basso."""
    inizio = time.perf_counter()
    migliore = None
    riusciti = 0
    valutazioni = 0
    for x0 in partenze:
        esito = esegui_bfgs_scipy(x0, f, nabla)
        valutazioni += esito["n_f"]
        if not np.all(np.isfinite(esito["x"])):
            continue
        riusciti += 1
        if migliore is None or esito["f"] < migliore["f"]:
            migliore = esito
    if migliore is None:
        return {
            "ok": False,
            "errore": "nessuna partenza ha prodotto un punto finito",
            "tempo": time.perf_counter() - inizio,
        }
    migliore = dict(migliore)
    migliore["tempo"] = time.perf_counter() - inizio
    migliore["riusciti"] = riusciti
    migliore["n_f"] = valutazioni
    migliore["ok"] = True
    return migliore


def esegui_basinhopping(f, nabla, x0, n_iter, stepsize):
    inizio = time.perf_counter()
    risultato = basinhopping(
        f,
        np.asarray(x0, dtype=float),
        niter=n_iter,
        stepsize=stepsize,
        seed=SEED,
        minimizer_kwargs={"method": "BFGS", "jac": nabla, "options": {"gtol": GTOL_SCIPY}},
    )
    x = np.asarray(risultato.x, dtype=float)
    return {
        "ok": np.all(np.isfinite(x)),
        "x": x,
        "f": float(risultato.fun),
        "grad": norma_gradiente(nabla, x),
        "tempo": time.perf_counter() - inizio,
        "n_f": int(risultato.nfev),
    }


def esegui_differential_evolution(f, bounds):
    inizio = time.perf_counter()
    risultato = differential_evolution(f, bounds, seed=SEED, polish=True)
    x = np.asarray(risultato.x, dtype=float)
    return {
        "ok": bool(risultato.success) and np.all(np.isfinite(x)),
        "x": x,
        "f": float(risultato.fun),
        "grad": None,
        "tempo": time.perf_counter() - inizio,
        "n_f": int(risultato.nfev),
        "messaggio": str(risultato.message),
    }


def stampa_sintesi(nome, esito, riferimento, f_star):
    print(f"  {nome}")
    if not esito["ok"] and "x" not in esito:
        print(f"    fallito dopo {fmt_tempo(esito['tempo'])}: {esito['errore']}")
        return
    print(
        f"    x = {fmt_x(esito['x'])}    f = {fmt_num(esito['f'])}"
        f"    f - f* = {fmt_num(esito['f'] - f_star)}"
    )
    print(
        f"    ||x - x*|| = {fmt_num(distanza(esito['x'], riferimento))}"
        f"    ||grad|| = {fmt_num(esito['grad']) if esito['grad'] is not None else 'n/d (metodo senza gradiente)'}"
    )
    extra = f"tempo {fmt_tempo(esito['tempo'])}"
    if "n_f" in esito:
        extra += f"  |  valutazioni f {esito['n_f']}"
    if "riusciti" in esito:
        extra += f"  |  partenze finite {esito['riusciti']}"
    print(f"    {extra}")


def confronta_multistart(
    nome, descrizione, f, nabla, riferimento, f_star, n_dim, bounds, stepsize, note_dominio, nota_se_mano_piu_basso=None
):
    separatore(f"MultiStart — {nome}")
    print(descrizione)
    print(note_dominio)
    partenze = partenze_come_multistart(N_PARTENZE, n_dim, SEED)
    print(f"{N_PARTENZE} partenze intere in {{1, ..., 10}}^{n_dim}, seed = {SEED}:")
    print("  " + ", ".join(fmt_x(x0, cifre=0) for x0 in partenze))

    print()
    print("Un BFGS per ogni partenza (stessi punti per i due metodi):")
    mano_piu_basso = 0
    scipy_piu_basso = 0
    punti_lontani = 0
    for x0 in partenze:
        mano = esegui_bfgs_mano(x0, f, nabla)
        scipy = esegui_bfgs_scipy(x0, f, nabla)
        if mano["ok"]:
            testo_mano = f"f={fmt_num(mano['f'])}  x={fmt_x(mano['x'])}"
        else:
            testo_mano = f"FALLITO ({mano['errore'].split(':', 1)[0]})"
        testo_scipy = f"f={fmt_num(scipy['f'])}  x={fmt_x(scipy['x'])}"
        print(f"  da {fmt_x(x0, cifre=0):<16} mano  {testo_mano}")
        print(f"  {'':<16}         scipy {testo_scipy}")
        if mano["ok"] and np.all(np.isfinite(scipy["x"])):
            if mano["f"] + 1e-2 < scipy["f"]:
                mano_piu_basso += 1
            elif scipy["f"] + 1e-2 < mano["f"]:
                scipy_piu_basso += 1
            if float(np.linalg.norm(mano["x"] - scipy["x"])) > 0.5:
                punti_lontani += 1

    print()
    print("Miglior punto tra le partenze, e due metodi globali di SciPy:")
    mano = esegui_multistart_mano(f, nabla, N_PARTENZE, n_dim, SEED)
    scipy_ms = esegui_multistart_scipy(f, nabla, partenze)
    salti = esegui_basinhopping(f, nabla, partenze[0], n_iter=N_PARTENZE, stepsize=stepsize)
    evoluzione = esegui_differential_evolution(f, bounds)
    evoluzione["grad"] = norma_gradiente(nabla, evoluzione["x"]) if evoluzione["ok"] else None

    stampa_sintesi("MultiStart a mano (il tuo)", mano, riferimento, f_star)
    stampa_sintesi("multistart con SciPy BFGS, stessi punti", scipy_ms, riferimento, f_star)
    stampa_sintesi(
        f"SciPy basinhopping (parte da {fmt_x(partenze[0], cifre=0)}, stepsize={stepsize}, salti={N_PARTENZE})",
        salti,
        riferimento,
        f_star,
    )
    stampa_sintesi(
        f"SciPy differential_evolution (dominio {fmt_bounds(bounds)})",
        evoluzione,
        riferimento,
        f_star,
    )

    if punti_lontani:
        print(
            f"  In {punti_lontani} partenze su {N_PARTENZE} i due BFGS finiscono a distanza > 0.5 "
            f"(tuo f più basso in {mano_piu_basso}, SciPy più basso in {scipy_piu_basso})."
        )
    if mano["ok"] and scipy_ms["ok"]:
        scarto_f = abs(mano["f"] - scipy_ms["f"])
        scarto_x = float(np.linalg.norm(np.asarray(mano["x"]) - np.asarray(scipy_ms["x"])))
        if scarto_f > 1e-2 and mano["f"] < scipy_ms["f"]:
            print(f"  Sul migliore, il tuo MultiStart scende più di SciPy: scarto di f = {fmt_num(scarto_f)}.")
            if nota_se_mano_piu_basso:
                print(f"  {nota_se_mano_piu_basso}")
        elif scarto_f > 1e-2:
            print(f"  Sul migliore, SciPy scende più del tuo MultiStart: scarto di f = {fmt_num(scarto_f)}.")
        elif scarto_x > 0.5:
            print(
                "  Il valore di f coincide, i punti no: su questa funzione ci sono più minimi con lo stesso valore."
            )
        else:
            print("  Sugli stessi punti, i due multistart trovano circa lo stesso minimo.")
    return {"mano": mano, "scipy": scipy_ms, "basinhopping": salti, "de": evoluzione}


def rifinisci(f, nabla, semi):
    """Porta dei punti già vicini a un minimo sul punto stazionario, con SciPy."""
    punti = []
    for seme in semi:
        risultato = minimize(
            f,
            np.asarray(seme, dtype=float),
            jac=nabla,
            method="BFGS",
            options={"gtol": 1e-12},
        )
        punti.append(np.asarray(risultato.x, dtype=float))
    return punti


# ---------------------------------------------------------------------------
# Grafici. Il percorso del BFGS a mano si ottiene registrando gli iterati
# della classe originale: il file BFGS.py non viene modificato.
# ---------------------------------------------------------------------------

COLORE_MANO = "#0072B2"
COLORE_SCIPY = "#D55E00"
COLORE_STELLA = "#F0C808"


class BFGSConPercorso(BFGS):
    """Stesso BFGS, con la lista dei punti accettati."""

    def __init__(self, starting_point, f, nabla_f, rho=0.5, c1=1e-4):
        super().__init__(starting_point, f, nabla_f, rho=rho, c1=c1)
        self.percorso = []

    def _bfgs(self, B_k, x_k, alpha, iteration):
        self.percorso.append(np.asarray(x_k, dtype=float).copy())
        x_finale = super()._bfgs(B_k, x_k, alpha, iteration)
        if iteration == 100:
            self.percorso.append(np.asarray(x_finale, dtype=float).copy())
        return x_finale


def percorso_mano(f, nabla, x0):
    algoritmo = BFGSConPercorso(np.asarray(x0, dtype=float), f, nabla)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            algoritmo.run()
    except Exception:
        if not algoritmo.percorso:
            raise
    punti = np.asarray(algoritmo.percorso, dtype=float)
    valori = np.array([float(f(x)) for x in punti])
    return punti, valori


def percorso_scipy_bfgs(f, nabla, x0):
    punti = [np.asarray(x0, dtype=float).copy()]

    def memorizza(xk):
        punti.append(np.asarray(xk, dtype=float).copy())

    minimize(
        f,
        np.asarray(x0, dtype=float),
        jac=nabla,
        method="BFGS",
        callback=memorizza,
        options={"gtol": GTOL_SCIPY},
    )
    arrivo = np.asarray(punti, dtype=float)
    valori = np.array([float(f(x)) for x in arrivo])
    return arrivo, valori


def etichetta_punto(x):
    pezzi = []
    for valore in np.asarray(x, dtype=float):
        if float(valore).is_integer():
            pezzi.append(str(int(valore)))
        else:
            pezzi.append(f"{valore:g}")
    return "(" + ", ".join(pezzi) + ")"


def finestra(percorsi, extra=(), margine=0.55):
    pezzi = [np.asarray(p, dtype=float) for p in percorsi if len(p)]
    if len(extra):
        pezzi.append(np.atleast_2d(np.asarray(extra, dtype=float)))
    punti = np.vstack(pezzi)
    return (
        (float(punti[:, 0].min() - margine), float(punti[:, 0].max() + margine)),
        (float(punti[:, 1].min() - margine), float(punti[:, 1].max() + margine)),
    )


def valuta_griglia(f, xlim, ylim, n=200):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    campioni = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.empty(campioni.shape[0])
    for i, punto in enumerate(campioni):
        Z[i] = f(punto)
    return X, Y, Z.reshape(X.shape)


def campo(Z, scala):
    if scala == "log":
        return np.log10(np.maximum(Z, 1e-8)), "log10(f)"
    return Z, "f"


def disegna_campo(ax, X, Y, Z, scala):
    valori, nome = campo(Z, scala)
    basso, alto = np.nanpercentile(valori, [1, 99])
    if not np.isfinite(basso) or alto <= basso:
        basso, alto = np.nanmin(valori), np.nanmax(valori)
    if alto <= basso:
        alto = basso + 1.0
    livelli = np.linspace(basso, alto, 18)
    riempimento = ax.contourf(X, Y, valori, levels=livelli, cmap="viridis", extend="both")
    ax.contour(X, Y, valori, levels=livelli, colors="black", linewidths=0.15, alpha=0.35)
    return riempimento, nome


def traccia(ax, punti, colore):
    ax.plot(punti[:, 0], punti[:, 1], color="white", lw=3.1, solid_capstyle="round", zorder=3)
    ax.plot(punti[:, 0], punti[:, 1], color=colore, lw=1.5, zorder=4)
    ax.scatter(
        punti[0, 0], punti[0, 1],
        s=42, facecolors="white", edgecolors=colore, linewidths=1.5, zorder=5,
    )
    ax.scatter(
        punti[-1, 0], punti[-1, 1],
        s=46, marker="X", color=colore, edgecolors="white", linewidths=0.5, zorder=6,
    )


def segna_minimi(ax, minimi):
    punti = np.atleast_2d(np.asarray(minimi, dtype=float))
    ax.scatter(
        punti[:, 0], punti[:, 1],
        marker="*", s=170, color=COLORE_STELLA, edgecolors="black", linewidths=0.45, zorder=7,
    )


def legenda_base():
    return [
        Line2D([0], [0], color=COLORE_MANO, lw=1.8, label="BFGS a mano"),
        Line2D([0], [0], color=COLORE_SCIPY, lw=1.8, label="SciPy BFGS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="partenza"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="black",
               markeredgecolor="black", markersize=7, label="arrivo"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORE_STELLA,
               markeredgecolor="black", markersize=12, label="minimo noto"),
    ]


def grafico_convergenza(cartella):
    fig, assi = plt.subplots(1, 3, figsize=(12.6, 3.9))
    casi = (
        ("Quadratica, da (0, 0)", quadratica, nabla_quadratica, np.array([0.0, 0.0])),
        ("Rosenbrock, da (−1.2, 1)", rosenbrock, nabla_rosenbrock, np.array([-1.2, 1.0])),
        ("Rastrigin, da (7, 7)", rastrigin, nabla_rastrigin, np.array([7.0, 7.0])),
    )
    for ax, (titolo, f, nabla, x0) in zip(assi, casi):
        _, valori_mano = percorso_mano(f, nabla, x0)
        _, valori_scipy = percorso_scipy_bfgs(f, nabla, x0)
        ax.semilogy(np.arange(len(valori_mano)), np.maximum(valori_mano, 1e-18), color=COLORE_MANO, label="BFGS a mano")
        ax.semilogy(np.arange(len(valori_scipy)), np.maximum(valori_scipy, 1e-18), color=COLORE_SCIPY, label="SciPy BFGS")
        ax.set_title(titolo)
        ax.set_xlabel("iterazione")
        ax.set_ylabel("f")
        ax.grid(True, which="both", alpha=0.28)
    assi[0].legend(fontsize=8)
    fig.suptitle("Il valore 0 esatto è disegnato a 1e−18, altrimenti la scala logaritmica non lo mostra", fontsize=9)
    fig.tight_layout()
    destinazione = cartella / "convergenza.png"
    fig.savefig(destinazione, dpi=150)
    plt.close(fig)
    return destinazione


def grafico_percorsi(cartella, nome, titolo, f, nabla, partenze, minimi, scala):
    tracce = []
    for x0 in partenze:
        tracce.append((percorso_mano(f, nabla, x0)[0], percorso_scipy_bfgs(f, nabla, x0)[0]))
    xlim, ylim = finestra([p for coppia in tracce for p in coppia], extra=minimi)
    X, Y, Z = valuta_griglia(f, xlim, ylim)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    riempimento, nome_barra = disegna_campo(ax, X, Y, Z, scala)
    for mano, scipy in tracce:
        traccia(ax, mano, COLORE_MANO)
        traccia(ax, scipy, COLORE_SCIPY)
    segna_minimi(ax, minimi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(titolo)
    ax.legend(handles=legenda_base(), fontsize=8, loc="best", framealpha=0.92)
    fig.colorbar(riempimento, ax=ax, fraction=0.046, pad=0.04, label=nome_barra)
    fig.tight_layout()
    destinazione = cartella / nome
    fig.savefig(destinazione, dpi=150)
    plt.close(fig)
    return destinazione


def grafico_pannelli_partenze(cartella, nome, titolo, f, nabla, partenze, minimi, scala):
    tracce = []
    for x0 in partenze:
        tracce.append((x0, percorso_mano(f, nabla, x0)[0], percorso_scipy_bfgs(f, nabla, x0)[0]))
    xlim, ylim = finestra(
        [p for _, mano, scipy in tracce for p in (mano, scipy)], extra=minimi, margine=0.9
    )
    X, Y, Z = valuta_griglia(f, xlim, ylim)
    fig, assi = plt.subplots(
        1, 2, figsize=(12.2, 6.5), sharex=True, sharey=True, layout="constrained"
    )
    colori = plt.cm.tab10(np.linspace(0, 0.9, len(tracce)))
    riempimento = None
    nome_barra = "f"
    for ax, indice, sottotitolo in (
        (assi[0], 1, "BFGS a mano"),
        (assi[1], 2, "SciPy BFGS"),
    ):
        riempimento, nome_barra = disegna_campo(ax, X, Y, Z, scala)
        for colore, (_, mano, scipy) in zip(colori, tracce):
            traccia(ax, mano if indice == 1 else scipy, colore)
        segna_minimi(ax, minimi)
        ax.set_title(sottotitolo)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    voci = [
        Line2D([0], [0], color=colori[i], lw=2, label=etichetta_punto(x0))
        for i, (x0, _, _) in enumerate(tracce)
    ]
    voci.append(
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORE_STELLA,
               markeredgecolor="black", markersize=12, label="minimi")
    )
    fig.legend(handles=voci, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    fig.colorbar(riempimento, ax=assi, shrink=0.82, label=nome_barra)
    fig.suptitle(titolo, fontsize=12)
    destinazione = cartella / nome
    fig.savefig(destinazione, dpi=150)
    plt.close(fig)
    return destinazione


def grafico_barre_rastrigin(cartella, tracce):
    indici = np.arange(len(tracce))
    larghezza = 0.38
    f_mano = [float(rastrigin(mano[-1])) for _, mano, _ in tracce]
    f_scipy = [float(rastrigin(scipy[-1])) for _, _, scipy in tracce]
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    ax.bar(indici - larghezza / 2, f_mano, larghezza, color=COLORE_MANO, label="BFGS a mano")
    ax.bar(indici + larghezza / 2, f_scipy, larghezza, color=COLORE_SCIPY, label="SciPy BFGS")
    for i, valore in enumerate(f_mano):
        ax.annotate(
            f"{valore:.0f}",
            (indici[i] - larghezza / 2, 0),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            va="bottom",
            color=COLORE_MANO,
            fontsize=8,
        )
    ax.set_xticks(indici)
    ax.set_xticklabels([etichetta_punto(x0) for x0, _, _ in tracce])
    ax.set_xlabel("partenza")
    ax.set_ylabel("f all'arrivo")
    ax.set_title("Rastrigin: da ogni intero il BFGS a mano arriva a f = 0")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    destinazione = cartella / "rastrigin_valori.png"
    fig.savefig(destinazione, dpi=150)
    plt.close(fig)
    return destinazione


def grafico_due_partenze(cartella, nome, titolo, f, nabla, partenze, minimi, scala):
    fig, assi = plt.subplots(1, len(partenze), figsize=(5.7 * len(partenze), 5.3))
    for ax, x0 in zip(assi, partenze):
        mano = percorso_mano(f, nabla, x0)[0]
        scipy = percorso_scipy_bfgs(f, nabla, x0)[0]
        xlim, ylim = finestra((mano, scipy), extra=minimi)
        X, Y, Z = valuta_griglia(f, xlim, ylim)
        riempimento, nome_barra = disegna_campo(ax, X, Y, Z, scala)
        traccia(ax, mano, COLORE_MANO)
        traccia(ax, scipy, COLORE_SCIPY)
        segna_minimi(ax, minimi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(f"da {etichetta_punto(x0)}")
        ax.legend(handles=legenda_base(), fontsize=7.5, loc="best", framealpha=0.92)
        fig.colorbar(riempimento, ax=ax, fraction=0.046, pad=0.04, label=nome_barra)
    fig.suptitle(titolo, fontsize=12)
    fig.tight_layout()
    destinazione = cartella / nome
    fig.savefig(destinazione, dpi=150)
    plt.close(fig)
    return destinazione


def salva_grafici():
    if plt is None:
        print("  matplotlib non è installato, i grafici non sono stati creati.")
        return []
    cartella = RADICE / "figure"
    cartella.mkdir(exist_ok=True)
    plt.rcParams.update({
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
    })

    creati = [grafico_convergenza(cartella)]
    creati.append(grafico_percorsi(
        cartella, "rosenbrock.png",
        "Rosenbrock, partenza (−1.2, 1)",
        rosenbrock, nabla_rosenbrock,
        (np.array([-1.2, 1.0]),),
        np.array([1.0, 1.0]),
        "log",
    ))
    creati.append(grafico_due_partenze(
        cartella, "rastrigin_percorsi.png",
        "Rastrigin",
        rastrigin, nabla_rastrigin,
        (np.array([7.0, 7.0]), np.array([7.4, 6.2])),
        np.zeros(2),
        "log",
    ))

    partenze = partenze_come_multistart(N_PARTENZE, 2, SEED)
    tracce_rastrigin = []
    for x0 in partenze:
        tracce_rastrigin.append((
            x0,
            percorso_mano(rastrigin, nabla_rastrigin, x0)[0],
            percorso_scipy_bfgs(rastrigin, nabla_rastrigin, x0)[0],
        ))
    creati.append(grafico_barre_rastrigin(cartella, tracce_rastrigin))

    minimi_pozzi = rifinisci(multi_pozzo, nabla_multi_pozzo, tuple(centro for centro, _ in POZZI))
    creati.append(grafico_pannelli_partenze(
        cartella, "multi_pozzo.png",
        "Multi-pozzo: stesse partenze, i due BFGS",
        multi_pozzo, nabla_multi_pozzo, partenze, minimi_pozzi, "lineare",
    ))
    minimi_himmelblau = rifinisci(
        himmelblau, nabla_himmelblau,
        (
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126]),
        ),
    )
    creati.append(grafico_pannelli_partenze(
        cartella, "himmelblau.png",
        "Himmelblau: f finale uguale, minimi diversi",
        himmelblau, nabla_himmelblau, partenze, minimi_himmelblau, "log",
    ))
    return creati


def main():
    np.set_printoptions(precision=5, suppress=True)
    print("Confronto didattico BFGS / MultiStart a mano  vs  SciPy")
    print("Gli avvisi numerici di NumPy sono nascosti dentro le chiamate al tuo BFGS;")
    print("se un punto esplode, la riga corrispondente riporta il fallimento.")

    controlla_gradienti()

    separatore("2. BFGS a mano  vs  scipy.optimize.minimize(method='BFGS')")
    print("Confronto a partenza singola. A entrambi viene passato lo stesso gradiente analitico.")
    print("SciPy può fermarsi appena il gradiente è piccolo; il tuo codice fa comunque 101 passi.")

    confronta_bfgs(
        "quadratica mal condizionata",
        "f(x) = 0.5 (x - x*)^T H (x - x*), con H = diag(100, 0.05). "
        "L'Hessiana iniziale del tuo BFGS è l'identità, molto diversa da H.",
        quadratica,
        nabla_quadratica,
        TARGET_QUADRATICA,
        0.0,
        (np.array([0.0, 0.0]), np.array([10.0, 10.0]), np.array([-5.0, 4.0])),
    )
    confronta_bfgs(
        "Rosenbrock",
        "La valle è lunga e piatta: il gradiente diventa piccolo prima ancora di arrivare in (1, 1).",
        rosenbrock,
        nabla_rosenbrock,
        np.array([1.0, 1.0]),
        0.0,
        (np.array([0.0, 0.0]), np.array([-1.2, 1.0]), np.array([2.0, 2.0])),
    )

    x_intero = np.array([7.0, 7.0])
    gradiente_intero = nabla_rastrigin(x_intero)
    direzione = -gradiente_intero
    arrivo = x_intero + 0.5 * direzione
    arrivo_txt = "l'origine" if np.linalg.norm(arrivo) < 1e-8 else fmt_x(arrivo)
    confronta_bfgs(
        "Rastrigin",
        "Qui ||x - x*|| è la distanza dal minimo globale, l'origine. "
        "Un minimo locale ha ||grad|| piccolo e ||x - x*|| grande: è il comportamento atteso di un metodo locale.\n"
        "Caso particolare delle partenze intere. In un intero il seno si annulla e ∇f(x) = 2x. "
        "Il tuo BFGS parte con B = I, quindi p = -∇f(x) = -2x. "
        f"Da {fmt_x(x_intero, cifre=0)} il primo tentativo è α = 1 e porta in -x, dove f è uguale a f(x): Armijo lo scarta. "
        f"Il tentativo dopo, α = 0.5, dà x + 0.5 p = {arrivo_txt}. "
        "f vale 0, quindi Armijo accetta e il metodo è già nel globale. "
        "Le decine di migliaia di valutazioni sono i passi successivi, a punto già fermo.\n"
        "SciPy usa Wolfe e un passo iniziale più corto, e si ferma nel locale vicino all'intero. "
        "Da una partenza non intera non atterri sull'origine, ma Armijo può comunque accettare un passo che cambia bacino. "
        "Il punto finale ha ||grad|| ≈ 0: è un minimo locale, spesso non quello di SciPy.",
        rastrigin,
        nabla_rastrigin,
        np.zeros(2),
        0.0,
        (x_intero, np.array([7.4, 6.2]), np.array([0.7, -0.4])),
    )

    minimi_himmelblau = rifinisci(
        himmelblau,
        nabla_himmelblau,
        (
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126]),
        ),
    )
    minimi_pozzi = rifinisci(
        multi_pozzo,
        nabla_multi_pozzo,
        tuple(centro for centro, _ in POZZI),
    )
    indice_pozzo_globale = int(np.argmin([multi_pozzo(p) for p in minimi_pozzi]))
    minimo_pozzo = minimi_pozzi[indice_pozzo_globale]
    f_pozzo = float(multi_pozzo(minimo_pozzo))

    separatore("3. MultiStart a mano  vs  metodi globali di SciPy")
    print("Il tuo MultiStart non riceve i punti: li genera dentro, con interi da 1 a 10.")
    print("Il multistart SciPy qui sotto usa esattamente la stessa sequenza (stesso seed).")
    print("basinhopping e differential_evolution sono un altro tipo di confronto:")
    print("non usano quelle partenze e, nel caso di Rastrigin, cercano anche fuori da [1, 10].")

    pozzi = confronta_multistart(
        "multi-pozzo, minimi di profondità diversa",
        "Tre buche gaussiane (sigma 1.5), profondità 3, 8 e 1.5. "
        "La sovrapposizione sposta un po' il fondo della buca più profonda rispetto al centro (7, 3). "
        f"Riferimento rifinito con SciPy: x* = {fmt_x(minimo_pozzo)}, f* = {fmt_num(f_pozzo)}.",
        multi_pozzo,
        nabla_multi_pozzo,
        minimo_pozzo,
        f_pozzo,
        n_dim=2,
        bounds=[(0.0, 11.0), (0.0, 11.0)],
        stepsize=1.0,
        note_dominio="Tutte le buche stanno dentro il cubo dei tuoi punti di partenza. "
        "Il multistart può trovare il globale solo se almeno una partenza cade nella buca giusta.",
    )
    himmel = confronta_multistart(
        "Himmelblau, quattro minimi globali con lo stesso valore",
        "f* = 0 in quattro punti. Dentro [1, 10]^2 c'è solo il minimo (3, 2); "
        "gli altri tre sono fuori, ma un BFGS non vincolato può comunque raggiungerli.",
        himmelblau,
        nabla_himmelblau,
        minimi_himmelblau,
        0.0,
        n_dim=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        stepsize=1.0,
        note_dominio="Qui tutti i minimi locali sono globali: arrivare in uno qualunque dà f = 0.",
    )
    rastrigin_2d = confronta_multistart(
        "Rastrigin in 2D, minimo globale nell'origine",
        "x* = (0, 0), f* = 0. Vicino a ogni intero diverso da zero c'è un minimo locale, con f circa ||x||^2.",
        rastrigin,
        nabla_rastrigin,
        np.array([0.0, 0.0]),
        0.0,
        n_dim=2,
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        stepsize=2.0,
        note_dominio="MultiStart pesca solo interi. Per il primo passo visto nella sezione BFGS, "
        "da ogni intero il tuo metodo cade sull'origine: ogni lancio trova già il globale, "
        "e il confronto tra i lanci non entra in gioco. "
        "Il multistart fatto con SciPy BFGS, sugli stessi interi, resta nei minimi locali. "
        "differential_evolution cerca in [-5.12, 5.12]^2.",
        nota_se_mano_piu_basso=(
            "Il multistart non ha cercato meglio: da un intero, Armijo accetta il salto nell'origine."
        ),
    )

    separatore("4. Rastrigin in 5D (la dimensione usata in multistart.py)")
    print("Stesso primo passo, in qualunque dimensione: da un vettore intero p = -2x e α = 1/2 portano all'origine.")
    print("Il tuo MultiStart parte solo da interi, quindi riporta f = 0 anche in 5D.")
    print("Il multistart con SciPy BFGS, sugli stessi punti, si ferma nei locali e f resta grande.")
    print("differential_evolution trova l'origine perché la cerca dentro [-5.12, 5.12]^5.")
    n_dim = 5
    partenze_5d = partenze_come_multistart(N_PARTENZE, n_dim, SEED)
    mano_5d = esegui_multistart_mano(rastrigin, nabla_rastrigin, N_PARTENZE, n_dim, SEED)
    scipy_5d = esegui_multistart_scipy(rastrigin, nabla_rastrigin, partenze_5d)
    origine = np.zeros(n_dim)
    stampa_sintesi("MultiStart a mano", mano_5d, origine, 0.0)
    stampa_sintesi("multistart SciPy BFGS, stessi punti", scipy_5d, origine, 0.0)
    print("  differential_evolution arriva all'origine cercando in [-5.12, 5.12]^5.")
    print("  Il tuo MultiStart ci arriva per il passo di Armijo. Quello con SciPy BFGS resta nei locali.")
    de_5d = esegui_differential_evolution(rastrigin, [(-5.12, 5.12)] * n_dim)
    de_5d["grad"] = norma_gradiente(nabla_rastrigin, de_5d["x"]) if de_5d["ok"] else None
    stampa_sintesi("SciPy differential_evolution", de_5d, origine, 0.0)

    separatore("5. Come leggere questo giro")
    print("BFGS.")
    print("  Sulla quadratica e su Rosenbrock c'è un solo minimo: i due punti devono coincidere e ||grad|| ≈ 0.")
    print("  Se lì il tuo ||grad|| resta grande, i 101 passi non sono bastati oppure B ha smesso di essere definita positiva.")
    print("  Il line search ricalcola f e il gradiente nel punto corrente a ogni dimezzamento di α,")
    print("  quindi le valutazioni restano alte anche quando il punto è già fermo.")
    print("  Su Rastrigin solo la partenza intera cade nell'origine. Dalle altre, Armijo cambia comunque bacino:")
    print("  ||grad|| ≈ 0 per entrambi, ma i punti non coincidono.")
    print()
    print("MultiStart.")
    print("  Tiene il f più piccolo tra i lanci. Non crea bacini nuovi.")
    if pozzi["mano"]["ok"]:
        print(
            f"  Multi-pozzo: il tuo metodo ha f = {fmt_num(pozzi['mano']['f'])}"
            f" (riferimento {fmt_num(f_pozzo)})."
        )
        print("  Nella tabella, la partenza che resta sulla buca con f circa -3 va scartata: il multistart tiene la più profonda.")
    if himmel["mano"]["ok"]:
        print(
            f"  Himmelblau: f = {fmt_num(himmel['mano']['f'])}. I quattro minimi valgono tutti zero,"
            " quindi punti lontani possono avere lo stesso f."
        )
    if rastrigin_2d["mano"]["ok"] and rastrigin_2d["scipy"]["ok"]:
        print(
            f"  Rastrigin: il tuo MultiStart ha f = {fmt_num(rastrigin_2d['mano']['f'])},"
            f" lo stesso schema fatto con SciPy ha f = {fmt_num(rastrigin_2d['scipy']['f'])}."
        )
        print("  Lo scarto viene dal line search di Armijo sulle partenze intere, visto sopra.")
    print()
    print("Per ripetere lo stesso sorteggio cambia SEED in cima al file. N_PARTENZE cambia il numero di lanci.")

    separatore("6. Grafici")
    print("I percorsi del BFGS a mano sono gli iterati davvero accettati, non una ricostruzione.")
    print("Il cerchio è la partenza, la croce l'arrivo, la stella il minimo noto.")
    for percorso in salva_grafici():
        print(f"  {percorso}")


if __name__ == "__main__":
    main()
