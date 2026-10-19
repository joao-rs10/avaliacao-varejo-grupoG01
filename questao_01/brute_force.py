"""Etapa A — Brute Force: ótimo exato de um subconjunto pequeno (benchmark do GA).

Enumera TODAS as combinações de níveis permitidos (produto cartesiano das listas
`permitidos`) e devolve a de maior fitness.

Duas implementações (ver otimizacao.py para a comparação antes/depois):
  * `brute_force_ingenuo`    — itertools.product + fitness recalculado por combinação
                               tempo O(M · n), M = Π m_i   (m_i = nº de níveis permitidos do SKU i)
  * `brute_force_vetorizado` — divide os SKUs em cabeça/cauda; a cauda é enumerada de uma
                               só vez com numpy (somas pré-computadas). tempo O(M) operações
                               elementares, memória O(Π m_cauda), não O(M).
"""
from __future__ import annotations

import itertools

import numpy as np

from .modelo import PENALIDADE_DESCONTO, ModeloQ1


def _combinacoes_total(modelo: ModeloQ1) -> int:
    return int(np.prod(modelo.n_opcoes.astype(object)))


def brute_force_ingenuo(modelo: ModeloQ1, limite_combinacoes: int = 5_000_000):
    """Versão simples/lenta: avalia o modelo completo de cada combinação."""
    total = _combinacoes_total(modelo)
    if total > limite_combinacoes:
        raise ValueError(f"{total} combinações excedem o limite {limite_combinacoes}.")
    melhor, melhor_pos = -np.inf, None
    for combo in itertools.product(*[range(int(m)) for m in modelo.n_opcoes]):
        f = modelo.fitness_ingenuo(combo)
        if f > melhor:
            melhor, melhor_pos = f, combo
    return np.array(melhor_pos), float(melhor), total


def brute_force_vetorizado(modelo: ModeloQ1, limite_combinacoes: int = 50_000_000):
    """Versão otimizada: mesma resposta exata, ordens de grandeza mais rápida."""
    total = _combinacoes_total(modelo)
    if total > limite_combinacoes:
        raise ValueError(f"{total} combinações excedem o limite {limite_combinacoes}.")
    n = modelo.n
    # cauda = últimos SKUs cujo produto de opções fica <= ~200k; cabeça = o resto
    corte, prod = n, 1
    while corte > 0 and prod * modelo.n_opcoes[corte - 1] <= 200_000:
        prod *= int(modelo.n_opcoes[corte - 1])
        corte -= 1
    cabeca, cauda = list(range(corte)), list(range(corte, n))

    # soma de valor/concessão de todas as combinações da cauda (broadcast, sem laços por combo)
    val_c = np.zeros(1)
    conc_c = np.zeros(1)
    for i in cauda:
        m = int(modelo.n_opcoes[i])
        val_c = (val_c[:, None] + modelo.val_pos[i, :m][None, :]).ravel()
        conc_c = (conc_c[:, None] + modelo.conc_pos[i, :m][None, :]).ravel()

    melhor, melhor_h, melhor_c = -np.inf, None, -1
    for combo_h in itertools.product(*[range(int(modelo.n_opcoes[i])) for i in cabeca]):
        vh = sum(modelo.val_pos[i, c] for i, c in zip(cabeca, combo_h))
        ch = sum(modelo.conc_pos[i, c] for i, c in zip(cabeca, combo_h))
        f = vh + val_c - PENALIDADE_DESCONTO * np.maximum(0.0, ch + conc_c - modelo.teto_desconto)
        j = int(np.argmax(f))
        if f[j] > melhor:
            melhor, melhor_h, melhor_c = float(f[j]), combo_h, j

    # reconstrói a combinação da cauda a partir do índice linear
    pos = np.zeros(n, dtype=int)
    for i, c in zip(cabeca, melhor_h):
        pos[i] = c
    resto = melhor_c
    for i in reversed(cauda):
        m = int(modelo.n_opcoes[i])
        pos[i] = resto % m
        resto //= m
    return pos, float(melhor), total
