"""Etapa B — Programação Dinâmica: trajetória semanal de preços por SKU.

Formulação (um problema por SKU; o acoplamento entre SKUs vem dos pesos de urgência):

  Estado     (t, s)  t = semana (0..T-1),  s = estoque remanescente (0..estoque_inicial_i)
  Decisão    k       nível de preço da semana t  (k = 0..K-1)
  Transição  s' = s - min(s, D),  D ~ distribuição de demanda de (i, t, k) — estimada por
             Monte Carlo e representada por Q nós de quantis  D_q  (q = 1..Q, prob. 1/Q)
  Recompensa w_t * margem_media(i,t,k) * min(s, D)
  Objetivo   V(t,s) = max_k  (1/Q) * Σ_q [ w_t * m(i,t,k) * min(s,D_q) + V(t+1, s - min(s,D_q)) ]
             V(T,s) = valor_residual_frac * custo_i * s

Os pesos w_t >= 1 dão mais valor ao caixa gerado ANTES de cada vencimento (a dívida é
paga com caixa acumulado); eles são calibrados no laço Monte Carlo <-> DP (ver main.py).

Bottom-Up: V(t+1, ·) é calculado UMA vez e reaproveitado por todos os (s, k, q) da semana t
(`n_consultas_V` consultas servidas por só `n_subproblemas` = T·(S+1) valores memoizados).
Como s' é fracionário, V(t+1, s') é obtido por interpolação linear entre estoques inteiros.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .modelo import ModeloQ2
from .monte_carlo import Estatisticas


@dataclass
class PlanoDP:
    politica: np.ndarray        # (I, T, Smax+1) int8 — nível ótimo por (SKU, semana, estoque)
    valor: list                 # valor[i] = matriz V(t, s) do SKU i, shape (T+1, S_i+1)
    n_subproblemas: int         # nº de estados (t,s) resolvidos (memoizados)
    n_consultas_V: int          # nº de consultas a V(t+1,·) servidas pela memoização


def pesos_de_urgencia(modelo: ModeloQ2, nu: dict | None = None) -> np.ndarray:
    """w_t = 1 + Σ_{vencimentos d >= t+1} nu_d  (semana t+1 conta para o vencimento d se t+1 <= d)."""
    w = np.ones(modelo.T)
    for d, val in (nu or {}).items():
        w[: int(d)] += val
    return w


def resolver_dp(modelo: ModeloQ2, stats: Estatisticas, pesos: np.ndarray | None = None) -> PlanoDP:
    """Tempo O(T·K·Q·ΣS_i); memória O(T·ΣS_i) para a política + O(S) por SKU para V."""
    T, K, I = modelo.T, modelo.K, modelo.I
    pesos = np.ones(T) if pesos is None else np.asarray(pesos, dtype=float)
    smax = int(modelo.estoque0.max())
    politica = np.zeros((I, T, smax + 1), dtype=np.int8)
    valores = []
    subproblemas = consultas = 0

    for i in range(I):
        S = int(modelo.estoque0[i])
        s = np.arange(S + 1, dtype=float)
        Vtab = np.zeros((T + 1, S + 1))
        Vtab[T] = modelo.p.valor_residual_frac * modelo.custo[i] * s       # V(T, s)
        V_prox = Vtab[T]
        for t in range(T - 1, -1, -1):
            melhor = np.full(S + 1, -np.inf)
            for k in range(K):
                Dq = stats.quantis_demanda[i, t, k]                          # (Q,)
                vendido = np.minimum(s[:, None], Dq[None, :])                # (S+1, Q)
                V_viz = np.interp((s[:, None] - vendido).ravel(), s, V_prox).reshape(vendido.shape)
                val = (pesos[t] * stats.margem_media[i, t, k] * vendido + V_viz).mean(axis=1)
                melhor_k = val > melhor
                melhor = np.where(melhor_k, val, melhor)
                politica[i, t, : S + 1][melhor_k] = k
                consultas += vendido.size
            V_prox = melhor
            Vtab[t] = melhor
            subproblemas += S + 1
        valores.append(Vtab)
    return PlanoDP(politica, valores, subproblemas, consultas)


def reconstruir_trajetoria(modelo: ModeloQ2, plano: PlanoDP, stats: Estatisticas) -> pd.DataFrame:
    """Caminho ótimo ESPERADO: a partir do estoque inicial aplica a política e desconta a venda
    esperada E[min(s, D)] de cada semana. Retorna uma linha por (SKU, semana)."""
    linhas = []
    for i in range(modelo.I):
        s = float(modelo.estoque0[i])
        for t in range(modelo.T):
            k = int(plano.politica[i, t, int(round(s))])
            Dq = stats.quantis_demanda[i, t, k]
            vend = float(np.minimum(s, Dq).mean())
            linhas.append({"sku": modelo.df["sku"].iat[i], "i": i, "semana": t + 1, "k": k,
                           "nivel_preco": float(modelo.niveis[k]),
                           "preco": float(modelo.precos[i, k]),
                           "estoque_inicio": s, "vendas_esperadas": vend,
                           "caixa_esperado": vend * float(stats.margem_media[i, t, k])})
            s -= vend
    return pd.DataFrame(linhas)
