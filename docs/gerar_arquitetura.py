"""Gera docs/arquitetura.png (fluxos das duas questões)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 6.5)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 6.5)

def caixa(x, y, w, h, txt, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=cor, ec="k"))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

def seta(x1, y1, x2, y2, txt="", rad=0.0):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.42, txt, ha="center", fontsize=7.5)

ax.text(6.5, 6.25, "Questão 1 — Operação Caixa 90", ha="center", fontweight="bold")
caixa(0.2, 4.6, 2.3, 1.1, "Dados Q1\n(csv + json)\nhash conferido", "#eeeeee")
caixa(3.2, 4.6, 2.6, 1.1, "A. Brute Force\n(subconjunto n>=6)\nótimo exato", "#cfe2f3")
caixa(6.5, 4.6, 2.6, 1.1, "B. Algoritmo Genético\n(catálogo completo)\npolítica de preços", "#d9ead3")
caixa(9.8, 4.6, 2.9, 1.1, "C. Greedy\norçamento promocional\nsobre a política do GA", "#fff2cc")
seta(2.5, 5.15, 3.2, 5.15); seta(5.8, 5.15, 6.5, 5.15, "gap % (valida o GA)"); seta(9.1, 5.15, 9.8, 5.15, "preços do GA")
seta(4.5, 4.55, 7.5, 4.55, "", rad=0.35)

ax.text(6.5, 3.35, "Questão 2 — Plano de Sobrevivência em 12 Semanas", ha="center", fontweight="bold")
caixa(0.2, 1.5, 2.3, 1.1, "Dados Q2\n(csv + json)", "#eeeeee")
caixa(3.0, 1.5, 2.6, 1.1, "A. Monte Carlo\n2000 cenários\nquantis de demanda,\nmargens por preço", "#cfe2f3")
caixa(6.2, 1.5, 2.6, 1.1, "B. Prog. Dinâmica\nestado (semana, estoque)\ntrajetória de preços", "#d9ead3")
caixa(9.4, 1.5, 2.0, 1.1, "C. Greedy semanal\nmídia/frete", "#fff2cc")
caixa(11.6, 0.2, 1.3, 1.0, "D. Monte Carlo\n5000 cenários\nindependentes", "#cfe2f3")
seta(2.5, 2.05, 3.0, 2.05); seta(5.6, 2.05, 6.2, 2.05, "estatísticas"); seta(8.8, 2.05, 9.4, 2.05, "plano de preços + V(t,s)")
seta(11.4, 1.7, 12.0, 1.2, "plano completo")
seta(11.6, 0.7, 4.3, 1.45, "calibração: pesos de urgência se P(pagar) < meta", rad=-0.25)
ax.text(6.5, 0.25, "P(pagar sem. 4, 8, 12) • caixa esperado • percentil de risco • estoque final", ha="center", fontsize=8, style="italic")
fig.tight_layout(); fig.savefig(Path(__file__).with_name("arquitetura.png"), dpi=140)
