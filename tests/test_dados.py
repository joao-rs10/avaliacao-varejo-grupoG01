import json
import shutil

import pandas as pd
import pytest

from common.dados import carregar_q1, carregar_q2, validar_catalogo, verificar_manifesto, COLUNAS_Q1
from conftest import RAIZ

DADOS = RAIZ / "data"


def test_manifesto_confere():
    m = verificar_manifesto(DADOS)
    assert m["grupo"] == "G01" and len(m["id_instancia"]) == 12


def test_alteracao_do_arquivo_quebra_hash(tmp_path):
    for f in DADOS.glob("*"):
        if f.suffix in (".csv", ".json"):
            shutil.copy(f, tmp_path / f.name)
    with open(tmp_path / "q1_catalogo_G01.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="hash"):
        verificar_manifesto(tmp_path)


def test_carrega_catalogos():
    d1, c1 = carregar_q1(DADOS)
    d2, c2 = carregar_q2(DADOS)
    assert len(d1) == 30 and len(d2) == 24
    assert c1["seed_final"] == c2["seed_final"]


def test_entradas_invalidas():
    d1, _ = carregar_q1(DADOS)
    with pytest.raises(ValueError):
        validar_catalogo(d1.drop(columns=["estoque"]), COLUNAS_Q1, "estoque", "demanda_base_90d")
    ruim = d1.copy(); ruim.loc[0, "estoque"] = 0
    with pytest.raises(ValueError):
        validar_catalogo(ruim, COLUNAS_Q1, "estoque", "demanda_base_90d")
    ruim = d1.copy(); ruim.loc[1, "sku"] = ruim.loc[0, "sku"]
    with pytest.raises(ValueError):
        validar_catalogo(ruim, COLUNAS_Q1, "estoque", "demanda_base_90d")
    with pytest.raises(ValueError):
        validar_catalogo(d1.iloc[0:0], COLUNAS_Q1, "estoque", "demanda_base_90d")
