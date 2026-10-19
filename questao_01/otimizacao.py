"""Otimização de código demonstrada por comparação antes/depois (Q1).

1) Brute Force: ingênuo (laço Python recalculando o modelo) vs vetorizado (cauda em numpy).
2) Fitness do GA: laço por indivíduo/SKU vs tabelas pré-calculadas + indexação numpy.
"""
from __future__ import annotations

import numpy as np

from common.medicao import medir

from .brute_force import brute_force_ingenuo, brute_force_vetorizado


def comparar_brute_force(modelo_pequeno):
    (pos_a, f_a, total), t_a, m_a = medir(brute_force_ingenuo, modelo_pequeno)
    (pos_b, f_b, _), t_b, m_b = medir(brute_force_vetorizado, modelo_pequeno)
    assert abs(f_a - f_b) < 1e-6, "as duas versões devem achar o mesmo ótimo"
    return {"combinacoes": total, "fitness_otimo": f_b,
            "antes_tempo_s": t_a, "antes_mem_mib": m_a,
            "depois_tempo_s": t_b, "depois_mem_mib": m_b,
            "speedup": t_a / t_b if t_b > 0 else float("inf")}


def comparar_fitness_ga(modelo, populacao=60, repeticoes=20, seed=0):
    rng = np.random.default_rng(seed)
    pop = np.floor(rng.random((populacao, modelo.n)) * modelo.n_opcoes[None, :]).astype(int)

    def antes():
        for _ in range(repeticoes):
            [modelo.fitness_ingenuo(ind) for ind in pop]

    def depois():
        for _ in range(repeticoes):
            modelo.fitness_pop(pop)

    _, t_a, m_a = medir(antes)
    _, t_b, m_b = medir(depois)
    return {"antes_tempo_s": t_a, "antes_mem_mib": m_a, "depois_tempo_s": t_b,
            "depois_mem_mib": m_b, "speedup": t_a / t_b if t_b > 0 else float("inf")}
