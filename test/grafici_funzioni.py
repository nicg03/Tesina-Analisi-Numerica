"""
Superfici 3D delle funzioni usate in test.py.

Sono tutte funzioni di due variabili, quindi il grafico è una superficie.
La quadratica dei test è già bidimensionale:

    f(x) = 0.5 (x - x*)^T H (x - x*),   H = diag(100, 0.05),   x* = (1, -2).

Rastrigin nei test compare anche in 5D: qui si disegna la versione in due
variabili, l'unica che sta in un grafico.

    python test/grafici_funzioni.py

Le figure finiscono in figure/, accanto agli algoritmi.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test import (
    POZZI,
    TARGET_QUADRATICA,
    himmelblau,
    multi_pozzo,
    quadratica,
    rastrigin,
    rosenbrock,
)

CARTELLA = Path(__file__).resolve().parent.parent / "figure"
COLORE_MINIMO = "#F0C808"

# Minimi globali di Himmelblau, tutti con f* = 0.
MINIMI_HIMMELBLAU = (
    np.array([3.0, 2.0]),
    np.array([-2.805118, 3.131312]),
    np.array([-3.779310, -3.283186]),
    np.array([3.584428, -1.848126]),
)


def griglia(f, xlim, ylim, n):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    punti = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([f(punto) for punto in punti], dtype=float)
    return X, Y, Z.reshape(X.shape)


def minimi_sulla_griglia(X, Y, Z, semi, raggio):
    """Punto più basso della griglia, in un intorno di ogni seme."""
    trovati = []
    for seme in semi:
        seme = np.asarray(seme, dtype=float)
        vicini = (X - seme[0]) ** 2 + (Y - seme[1]) ** 2 <= raggio ** 2
        if not np.any(vicini):
            trovati.append((float(seme[0]), float(seme[1]), float("nan")))
            continue
        indici = np.argwhere(vicini)
        scelto = indici[np.argmin(Z[vicini])]
        i, j = int(scelto[0]), int(scelto[1])
        trovati.append((float(X[i, j]), float(Y[i, j]), float(Z[i, j])))
    return trovati


def quota_log(z):
    return float(np.log10(1.0 + max(float(z), 0.0)))


def disegna(
    nome, titolo, f, xlim, ylim, minimi,
    n=170, elev=28, azim=-58, scala="lineare", contorni=True, n_livelli=14,
    altezza_stella=0.55,
):
    X, Y, Z = griglia(f, xlim, ylim, n)
    punti = minimi(X, Y, Z) if callable(minimi) else list(minimi)
    if scala == "log1p":
        Z_dis = np.log10(1.0 + np.maximum(Z, 0.0))
        nome_asse = r"$\log_{10}(1+f)$"
    else:
        Z_dis = Z
        nome_asse = r"$f$"

    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    superficie = ax.plot_surface(
        X, Y, Z_dis,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        shade=True,
    )

    z_basso = float(np.min(Z_dis))
    z_alto = float(np.max(Z_dis))
    span = z_alto - z_basso
    if span <= 0.0:
        span = 1.0
    offset = z_basso - 0.12 * span
    if contorni:
        ax.contour(
            X, Y, Z_dis,
            zdir="z",
            offset=offset,
            levels=n_livelli,
            cmap="viridis",
            linewidths=0.7,
        )

    for x_min, y_min, z_min in punti:
        if not np.isfinite(z_min):
            z_min = float(f(np.array([x_min, y_min])))
        z_punto = quota_log(z_min) if scala == "log1p" else float(z_min)
        z_testa = z_basso + altezza_stella * span
        if z_testa < z_punto + 0.06 * span:
            z_testa = z_punto + 0.12 * span
        ax.plot(
            [x_min, x_min], [y_min, y_min], [z_punto, z_testa],
            color="black", lw=0.8,
        )
        ax.scatter(
            [x_min], [y_min], [z_testa],
            marker="*",
            s=180,
            color=COLORE_MINIMO,
            edgecolors="black",
            linewidths=0.6,
            depthshade=False,
        )

    lato = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.set_box_aspect((
        (xlim[1] - xlim[0]) / lato,
        (ylim[1] - ylim[0]) / lato,
        0.62,
    ))
    ax.set_zlim(offset, z_alto + 0.04 * span)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$x_1$", labelpad=8)
    ax.set_ylabel(r"$x_2$", labelpad=8)
    ax.set_zlabel(nome_asse, labelpad=8)
    ax.set_title(titolo, pad=12)
    fig.colorbar(superficie, ax=ax, shrink=0.62, pad=0.08, label=nome_asse)
    fig.tight_layout()

    destinazione = CARTELLA / nome
    fig.savefig(destinazione, dpi=160)
    plt.close(fig)
    return destinazione


def main():
    CARTELLA.mkdir(exist_ok=True)
    plt.rcParams.update({
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
    })

    creati = []
    creati.append(disegna(
        "superficie_quadratica.png",
        "",
        quadratica,
        (-2.0, 4.0),
        (-8.0, 4.0),
        lambda X, Y, Z: minimi_sulla_griglia(X, Y, Z, (TARGET_QUADRATICA,), raggio=1.5),
        elev=38,
        azim=-64,
        altezza_stella=0.16,
    ))
    creati.append(disegna(
        "superficie_rosenbrock.png",
        "",
        rosenbrock,
        (-2.0, 2.0),
        (-1.0, 3.0),
        lambda X, Y, Z: minimi_sulla_griglia(X, Y, Z, (np.array([1.0, 1.0]),), raggio=0.8),
        elev=48,
        azim=-115,
        scala="log1p",
        altezza_stella=0.42,
    ))
    creati.append(disegna(
        "superficie_multi_pozzo.png",
        "",
        multi_pozzo,
        (0.0, 11.0),
        (0.0, 11.0),
        lambda X, Y, Z: minimi_sulla_griglia(
            X, Y, Z, tuple(centro for centro, _ in POZZI), raggio=1.6
        ),
        elev=52,
        azim=-125,
        altezza_stella=0.9,
    ))
    creati.append(disegna(
        "superficie_himmelblau.png",
        "",
        himmelblau,
        (-5.0, 5.0),
        (-5.0, 5.0),
        lambda X, Y, Z: minimi_sulla_griglia(X, Y, Z, MINIMI_HIMMELBLAU, raggio=1.2),
        elev=38,
        azim=-55,
        scala="log1p",
    ))
    creati.append(disegna(
        "superficie_rastrigin.png",
        "",
        rastrigin,
        (-5.12, 5.12),
        (-5.12, 5.12),
        lambda X, Y, Z: minimi_sulla_griglia(X, Y, Z, (np.zeros(2),), raggio=0.8),
        elev=38,
        azim=-60,
        contorni=False,
        altezza_stella=0.28,
    ))

    for percorso in creati:
        print(percorso)


if __name__ == "__main__":
    main()
