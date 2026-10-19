"""Etapa A (e D) — Monte Carlo: estatísticas de demanda/caixa por preço e validação do plano.

Papéis:
  * `estatisticas_por_preco`  (A) — alimenta a Programação Dinâmica com, para cada
        (SKU, semana, nível de preço): nós de quantis da demanda e margem unitária média.
  * `simular_plano`           (D) — valida o plano completo (DP + Greedy) em cenários
        INDEPENDENTES dos usados na construção (outro fluxo de números aleatórios).

Reprodutibilidade: cada uso tem seu próprio gerador `np.random.default_rng(SeedSequence([seed, TAG]))`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.incentivo import uplift

from .modelo import ModeloQ2

TAG_CONSTRUCAO = 301   # cenários usados para estimar estatísticas e calibrar o plano
TAG_VALIDACAO = 302    # cenários independentes usados só na validação final


def rng_para(seed: int, tag: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(tag)]))


@dataclass
class Choques:
    eps_demanda: np.ndarray   # (n, T, I) média 1
    fator_conc: np.ndarray    # (n, T, I) pc_t / pc_0
    fator_log: np.ndarray     # (n, T, I) média 1


def gerar_choques(modelo: ModeloQ2, n: int, rng: np.random.Generator) -> Choques:
    """Três fontes de incerteza: demanda, preço do concorrente, custo logístico."""
    if n < 1:
        raise ValueError("n de cenários deve ser >= 1")
    T, I, p = modelo.T, modelo.I, modelo.p

    def correlacionado():
        zc = rng.standard_normal((n, T, 1))
        zi = rng.standard_normal((n, T, I))
        return np.sqrt(p.rho) * zc + np.sqrt(1.0 - p.rho) * zi

    z_d, z_c, z_l = correlacionado(), correlacionado(), correlacionado()
    eps = np.exp(p.sigma_demanda * z_d - 0.5 * p.sigma_demanda ** 2)
    fator_conc = np.exp(np.cumsum(p.sigma_concorrente * z_c, axis=1))   # passeio aleatório
    fator_log = np.exp(p.sigma_logistico * z_l - 0.5 * p.sigma_logistico ** 2)
    return Choques(eps, fator_conc, fator_log)


def demanda(modelo: ModeloQ2, eps, fator_conc, fator_preco_k):
    """Demanda semanal (broadcast sobre cenários). fator_preco_k tem shape (I,) ou (n,I)."""
    return modelo.base * eps * fator_preco_k * fator_conc ** modelo.p.theta


@dataclass
class Estatisticas:
    quantis_demanda: np.ndarray   # Dq (I, T, K, Q)
    demanda_media: np.ndarray     # (I, T, K)
    demanda_desvio: np.ndarray    # (I, T, K)
    margem_media: np.ndarray      # (I, T, K)  margem unitária média (já líquida do frete)
    n_cenarios: int


def estatisticas_por_preco(modelo: ModeloQ2, n: int, seed: int) -> Estatisticas:
    """Monte Carlo (construção). Tempo O(K·n·T·I + K·T·I·n·log n); memória O(n·T·I)."""
    rng = rng_para(seed, TAG_CONSTRUCAO)
    ch = gerar_choques(modelo, n, rng)
    Q = modelo.p.n_quantis
    qs = (np.arange(Q) + 0.5) / Q
    Dq = np.empty((modelo.I, modelo.T, modelo.K, Q))
    Dm = np.empty((modelo.I, modelo.T, modelo.K))
    Ds = np.empty_like(Dm)
    Mm = np.empty_like(Dm)
    frete_medio = (modelo.frete[None, :] * ch.fator_log).mean(axis=0)           # (T, I)
    for k in range(modelo.K):
        D = demanda(modelo, ch.eps_demanda, ch.fator_conc, modelo.fator_preco[:, k])   # (n,T,I)
        q = np.quantile(D, qs, axis=0)                                           # (Q,T,I)
        Dq[:, :, k, :] = np.transpose(q, (2, 1, 0))
        Dm[:, :, k] = D.mean(axis=0).T
        Ds[:, :, k] = D.std(axis=0).T
        Mm[:, :, k] = modelo.margem_det[:, k][:, None] - frete_medio.T
    return Estatisticas(Dq, Dm, Ds, Mm, n)


@dataclass
class PlanoPromo:
    gasto: np.ndarray          # (T, I)  R$ investidos por SKU/semana
    fator_demanda: np.ndarray  # (T, I)  1 + uplift


def plano_sem_promo(modelo: ModeloQ2) -> PlanoPromo:
    return PlanoPromo(np.zeros((modelo.T, modelo.I)), np.ones((modelo.T, modelo.I)))


def politica_constante(modelo: ModeloQ2, nivel_idx: int) -> np.ndarray:
    smax = int(modelo.estoque0.max())
    return np.full((modelo.I, modelo.T, smax + 1), nivel_idx, dtype=np.int8)


def simular_plano(modelo: ModeloQ2, politica: np.ndarray, promo: PlanoPromo, n: int, seed: int,
                  tag: int = TAG_VALIDACAO, skus_registrar=None, choques: Choques | None = None) -> dict:
    """Simula o plano completo (política de preço dependente do estoque + promoções).

    politica: array (I, T, Smax+1) com o índice de nível de preço por (SKU, semana, estoque).
    Tempo O(n·T·I); memória O(n·T·I).
    """
    p = modelo.p
    ch = choques or gerar_choques(modelo, n, rng_para(seed, tag))
    n = ch.eps_demanda.shape[0]
    I, T = modelo.I, modelo.T
    ar = np.arange(I)[None, :]
    estoque = np.tile(modelo.estoque0.astype(float), (n, 1))
    saldo = np.full(n, p.caixa_inicial)
    atraso = np.zeros(n)
    receita = np.zeros(n)
    caixa_op = np.zeros(n)
    venc = dict(modelo.vencimentos)
    ok = np.zeros((n, len(venc)), dtype=bool)
    saldo_no_venc = np.zeros((n, len(venc)))
    ordem_venc = {s: j for j, s in enumerate(sorted(venc))}
    rec = None
    if skus_registrar is not None:
        rec = {"nivel": np.zeros((n, T, len(skus_registrar)), dtype=np.int8),
               "estoque": np.zeros((n, T + 1, len(skus_registrar)))}
        rec["estoque"][:, 0, :] = estoque[:, skus_registrar]

    for t in range(T):
        s_idx = np.rint(estoque).astype(int)
        k = politica[ar, t, s_idx].astype(int)                                   # (n,I)
        preco = modelo.precos[ar, k]
        m_un = modelo.margem_det[ar, k] - modelo.frete[None, :] * ch.fator_log[:, t, :]
        D = demanda(modelo, ch.eps_demanda[:, t, :], ch.fator_conc[:, t, :], modelo.fator_preco[ar, k])
        D = D * promo.fator_demanda[t][None, :]
        vendido = np.minimum(estoque, D)
        fluxo = (vendido * m_un).sum(axis=1) - promo.gasto[t].sum()
        caixa_op += fluxo
        receita += (vendido * preco).sum(axis=1)
        saldo += fluxo
        estoque -= vendido
        if (t + 1) in venc:
            devido = venc[t + 1] + atraso
            paga = saldo >= devido
            j = ordem_venc[t + 1]
            ok[:, j] = paga
            saldo_no_venc[:, j] = saldo
            saldo = np.where(paga, saldo - devido, saldo)
            atraso = np.where(paga, 0.0, devido)
        if rec is not None:
            rec["nivel"][:, t, :] = k[:, skus_registrar]
            rec["estoque"][:, t + 1, :] = estoque[:, skus_registrar]

    return {"ok": ok, "caixa_final": saldo, "caixa_operacional": caixa_op, "receita": receita,
            "estoque_final": estoque.sum(axis=1), "saldo_no_vencimento": saldo_no_venc,
            "vencimentos": sorted(venc), "registro": rec, "n": n}


def resumir(res: dict, modelo: ModeloQ2) -> dict:
    ok = res["ok"]
    n = res["n"]
    todos = ok.all(axis=1)
    cf = res["caixa_final"]
    est0 = float(modelo.estoque0.sum())
    return {
        "n_cenarios": int(n),
        "prob_pagamento_por_vencimento": {f"semana_{s}": float(ok[:, j].mean())
                                          for j, s in enumerate(res["vencimentos"])},
        "prob_pagar_todos_no_prazo": float(todos.mean()),
        "cenarios_de_falha": int((~todos).sum()),
        "taxa_de_falha": float((~todos).mean()),
        "caixa_final_esperado": float(cf.mean()),
        "caixa_final_desvio": float(cf.std()),
        f"caixa_final_p{modelo.p.percentil_risco:g}": float(np.percentile(cf, modelo.p.percentil_risco)),
        "caixa_final_p50": float(np.percentile(cf, 50)),
        "caixa_final_p95": float(np.percentile(cf, 95)),
        "caixa_operacional_esperado": float(res["caixa_operacional"].mean()),
        "receita_esperada": float(res["receita"].mean()),
        "margem_esperada": float(res["caixa_operacional"].mean() / res["receita"].mean()),
        "estoque_final_medio_un": float(res["estoque_final"].mean()),
        "estoque_final_pct_do_inicial": float(100 * res["estoque_final"].mean() / est0),
        "atende_limite_de_risco": bool((1 - todos.mean()) <= modelo.p.limite_risco_falha),
    }
