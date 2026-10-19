"""Cenário-surpresa por parâmetros: toda alteração entra por dados/configuração,
nunca por edição do algoritmo (ver README, seção 'Cenário-surpresa')."""
from __future__ import annotations

import copy


def _remover(df, skus, col="sku"):
    if not skus:
        return df
    faltando = [s for s in skus if s not in set(df[col])]
    if faltando:
        raise ValueError(f"SKU(s) inexistente(s) para remover: {faltando}")
    out = df[~df[col].isin(skus)].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("Remover esses SKUs deixaria o catálogo vazio.")
    return out


def aplicar_cenario_q1(df, cfg, meta_mult=1.0, orcamento_mult=1.0, custo_logistico_mult=1.0,
                       remover_skus=None, margem_minima=None):
    df, cfg = df.copy(), copy.deepcopy(cfg)
    cfg["meta_caixa_90d"] *= meta_mult
    cfg["orcamento_promocional"] *= orcamento_mult
    df["custo_logistico"] = df["custo_logistico"] * custo_logistico_mult
    if margem_minima is not None:
        cfg["margem_minima_percentual"] = margem_minima
    return _remover(df, remover_skus), cfg


def aplicar_cenario_q2(df, cfg, preco_concorrente_mult=1.0, custo_logistico_mult=1.0,
                       orcamento_mult=1.0, divida_mult=1.0, remover_skus=None):
    df, cfg = df.copy(), copy.deepcopy(cfg)
    df["preco_concorrente_base"] = df["preco_concorrente_base"] * preco_concorrente_mult
    df["custo_logistico_base"] = df["custo_logistico_base"] * custo_logistico_mult
    cfg["orcamento_promocional_semanal"] *= orcamento_mult
    cfg["dividas"] = {k: v * divida_mult for k, v in cfg["dividas"].items()}
    return _remover(df, remover_skus), cfg
