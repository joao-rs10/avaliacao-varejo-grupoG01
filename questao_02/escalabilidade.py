"""Escalabilidade em tempo e memória: 3 tamanhos de entrada (nº de SKUs) e 3 nº de cenários."""
from __future__ import annotations

import pandas as pd

from common.medicao import medir

from .dynamic_programming import reconstruir_trajetoria, resolver_dp
from .greedy import greedy_semanal
from .modelo import ModeloQ2
from .monte_carlo import estatisticas_por_preco, simular_plano


def medir_etapas(modelo: ModeloQ2, seed: int, n_cons: int, n_val: int) -> dict:
    stats, t_mc, m_mc = medir(estatisticas_por_preco, modelo, n_cons, seed)
    dp, t_dp, m_dp = medir(resolver_dp, modelo, stats)
    traj = reconstruir_trajetoria(modelo, dp, stats)
    promo, t_g, m_g = medir(greedy_semanal, modelo, traj, stats, dp)
    _, t_v, m_v = medir(simular_plano, modelo, dp.politica, promo, n_val, seed)
    return {"n_skus": modelo.I, "n_cenarios_construcao": n_cons, "n_cenarios_validacao": n_val,
            "soma_estoques": int(modelo.estoque0.sum()),
            "mc_tempo_s": t_mc, "mc_mem_mib": m_mc, "dp_tempo_s": t_dp, "dp_mem_mib": m_dp,
            "greedy_tempo_s": t_g, "greedy_mem_mib": m_g, "validacao_tempo_s": t_v, "validacao_mem_mib": m_v}


def estudo_escalabilidade(modelo: ModeloQ2, seed: int, tamanhos=(6, 12, 24), cenarios=(2000, 3500, 5000)):
    linhas = []
    for n in tamanhos:                                   # varia o nº de SKUs (cenários fixos)
        sub = modelo.subconjunto(min(n, modelo.I))
        linhas.append({"eixo": "n_skus", **medir_etapas(sub, seed, 2000, 5000)})
    for c in cenarios:                                   # varia o nº de cenários (SKUs fixos)
        linhas.append({"eixo": "n_cenarios", **medir_etapas(modelo, seed, max(c, 2000), max(c, 5000))})
    return pd.DataFrame(linhas)
