"""Modelo de incentivo comercial (cupom / cashback / frete subsidiado), comum às duas questões.

Investir x reais em um SKU eleva a demanda em  uplift(x) = a * (1 - exp(-x / k)):
retorno marginal decrescente (côncavo) com teto `a`. Como o ganho é côncavo, a
alocação gulosa por maior benefício marginal por real é ótima para esse modelo
(problema da mochila separável côncavo, com passo discreto).
"""
from __future__ import annotations

import heapq

import numpy as np


def uplift(x, a: float, k):
    """Aumento relativo de demanda com investimento x (R$) — vetorizável."""
    return a * (1.0 - np.exp(-np.asarray(x, dtype=float) / np.asarray(k, dtype=float)))


def alocar_guloso(ganho_fn, n_itens: int, orcamento: float, passo: float,
                  limiar_retorno: float = 1.0):
    """Greedy por benefício marginal por real investido.

    ganho_fn(i, x) -> ganho bruto de caixa do item i ao investir x (côncava em x).
    A cada iteração investe `passo` no item de maior (Δganho / passo) enquanto o
    retorno marginal for > `limiar_retorno` (1.0 = cada R$ investido devolve > R$ 1)
    e houver orçamento. Complexidade: O(n + (B/passo) log n).
    Retorna (x[n], n_iteracoes).
    """
    if orcamento < 0 or passo <= 0:
        raise ValueError("orçamento deve ser >= 0 e passo > 0")
    x = np.zeros(n_itens)
    heap = []
    for i in range(n_itens):
        r = (ganho_fn(i, passo) - ganho_fn(i, 0.0)) / passo
        heapq.heappush(heap, (-r, i))
    restante = float(orcamento)
    it = 0
    while heap and restante >= passo - 1e-9:
        neg_r, i = heapq.heappop(heap)
        if -neg_r <= limiar_retorno:
            break  # o melhor retorno marginal já não compensa
        x[i] += passo
        restante -= passo
        it += 1
        r = (ganho_fn(i, x[i] + passo) - ganho_fn(i, x[i])) / passo
        heapq.heappush(heap, (-r, i))
    return x, it
