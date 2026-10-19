"""Leitura, validação e verificação de integridade dos arquivos oficiais da instância.

Os arquivos CSV/JSON gerados pelo notebook oficial NUNCA são editados aqui:
o código apenas lê e confere os hashes SHA-256 registrados no manifesto.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

COLUNAS_Q1 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque",
    "demanda_base_90d", "elasticidade", "taxa_marketplace",
    "custo_logistico", "idade_estoque_dias",
]
COLUNAS_Q2 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial",
    "demanda_base_semanal", "elasticidade", "taxa_marketplace",
    "custo_logistico_base", "preco_concorrente_base",
]


def sha256_arquivo(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def descobrir_grupo(pasta) -> str:
    """Lê o código do grupo (ex.: 'G01') do manifesto_dados.json da pasta."""
    manifesto = Path(pasta) / "manifesto_dados.json"
    if not manifesto.exists():
        raise FileNotFoundError(f"manifesto_dados.json não encontrado em {pasta}")
    with manifesto.open(encoding="utf-8") as f:
        return json.load(f)["grupo"]


def verificar_manifesto(pasta, grupo: str | None = None) -> dict:
    """Recalcula os hashes dos 4 arquivos e compara com o manifesto.

    Retorna o manifesto; levanta ValueError se algum arquivo foi alterado.
    """
    pasta = Path(pasta)
    grupo = grupo or descobrir_grupo(pasta)
    with (pasta / "manifesto_dados.json").open(encoding="utf-8") as f:
        manifesto = json.load(f)
    erros = []
    for nome, hash_esperado in manifesto["arquivos"].items():
        arq = pasta / nome
        if not arq.exists():
            erros.append(f"{nome}: arquivo ausente")
        elif sha256_arquivo(arq) != hash_esperado:
            erros.append(f"{nome}: hash difere do manifesto (arquivo alterado)")
    if erros:
        raise ValueError("Integridade da instância violada:\n" + "\n".join(erros))
    conjunto = hashlib.sha256(
        "".join(manifesto["arquivos"][k] for k in sorted(manifesto["arquivos"])).encode("utf-8")
    ).hexdigest()
    if conjunto != manifesto["sha256_conjunto"]:
        raise ValueError("sha256_conjunto do manifesto não confere.")
    return manifesto


def validar_catalogo(df: pd.DataFrame, colunas: list[str], col_estoque: str, col_demanda: str) -> None:
    """Validações de entrada (usadas também nos testes de entradas inválidas)."""
    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no catálogo: {faltando}")
    if len(df) == 0:
        raise ValueError("Catálogo vazio.")
    if df["sku"].duplicated().any():
        raise ValueError("SKU duplicado no catálogo.")
    if (df["custo_unitario"] <= 0).any() or (df["preco_atual"] <= 0).any():
        raise ValueError("Custo e preço devem ser positivos.")
    if (df[col_estoque] <= 0).any() or (df[col_demanda] <= 0).any():
        raise ValueError("Estoque e demanda devem ser positivos.")
    if (df["elasticidade"] <= 0).any():
        raise ValueError("Elasticidade deve ser positiva.")
    if not df["taxa_marketplace"].between(0, 1).all():
        raise ValueError("taxa_marketplace fora de [0, 1].")


def carregar_q1(pasta, grupo: str | None = None, verificar: bool = True):
    pasta = Path(pasta)
    grupo = grupo or descobrir_grupo(pasta)
    if verificar:
        verificar_manifesto(pasta, grupo)
    df = pd.read_csv(pasta / f"q1_catalogo_{grupo}.csv", encoding="utf-8-sig")
    with (pasta / f"q1_config_{grupo}.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    validar_catalogo(df, COLUNAS_Q1, "estoque", "demanda_base_90d")
    return df, cfg


def carregar_q2(pasta, grupo: str | None = None, verificar: bool = True):
    pasta = Path(pasta)
    grupo = grupo or descobrir_grupo(pasta)
    if verificar:
        verificar_manifesto(pasta, grupo)
    df = pd.read_csv(pasta / f"q2_catalogo_{grupo}.csv", encoding="utf-8-sig")
    with (pasta / f"q2_config_{grupo}.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    validar_catalogo(df, COLUNAS_Q2, "estoque_inicial", "demanda_base_semanal")
    return df, cfg
