import numpy as np
import pandas as pd
import pytest

from common.cenarios import aplicar_cenario_q1
from common.dados import carregar_q1
from common.incentivo import alocar_guloso
from questao_01.brute_force import brute_force_ingenuo, brute_force_vetorizado
from questao_01.genetic_algorithm import CONFIGS_PADRAO, ConfigGA, algoritmo_genetico
from questao_01.greedy import greedy_incentivos
from questao_01.main import selecionar_subconjunto
from questao_01.modelo import ModeloQ1, ParametrosQ1
from conftest import RAIZ


@pytest.fixture(scope="module")
def modelo():
    df, cfg = carregar_q1(RAIZ / "data")
    return ModeloQ1(df, ParametrosQ1.de_config(cfg))


def mini(modelo, n=6):
    return modelo.subconjunto(selecionar_subconjunto(modelo, n, 1))


def test_margem_minima_respeitada(modelo):
    pos = modelo.politica_base()
    k = modelo.niveis_da_politica(pos)
    ar = np.arange(modelo.n)
    ativos = ~modelo.retirado
    pct = modelo.margem_un[ar, k] / modelo.precos[ar, k]
    assert (pct[ativos] >= modelo.p.margem_minima - 1e-12).all()


def test_sku_sem_nivel_viavel_e_retirado(modelo):
    assert modelo.retirado.sum() == 3
    assert (modelo.unidades[modelo.retirado] == 0).all()


def test_vendas_nao_excedem_estoque(modelo):
    assert (modelo.unidades <= modelo.estoque[:, None]).all()


def test_fitness_vetorizado_igual_ingenuo(modelo):
    rng = np.random.default_rng(0)
    for _ in range(5):
        pos = np.floor(rng.random(modelo.n) * modelo.n_opcoes).astype(int)
        assert modelo.fitness(pos) == pytest.approx(modelo.fitness_ingenuo(pos))


def test_brute_force_ingenuo_igual_vetorizado(modelo):
    sub = mini(modelo, 6)
    _, f1, t1 = brute_force_ingenuo(sub)
    _, f2, t2 = brute_force_vetorizado(sub)
    assert f1 == pytest.approx(f2) and t1 == t2


def test_brute_force_limite(modelo):
    with pytest.raises(ValueError):
        brute_force_vetorizado(modelo, limite_combinacoes=10)


def test_ga_reprodutivel_por_seed(modelo):
    cfg = ConfigGA(populacao=20, geracoes=15, paciencia=100)
    a = algoritmo_genetico(modelo, cfg, 5)
    b = algoritmo_genetico(modelo, cfg, 5)
    assert (a[0] == b[0]).all() and a[1] == b[1] and a[2]["melhor"] == b[2]["melhor"]


def test_ga_elitismo_monotono(modelo):
    _, _, h = algoritmo_genetico(modelo, ConfigGA(populacao=20, geracoes=30, paciencia=100, elite=2), 3)
    assert all(b >= a - 1e-9 for a, b in zip(h["melhor"], h["melhor"][1:]))


def test_ga_chega_ao_otimo_no_subconjunto(modelo):
    sub = mini(modelo, 6)
    _, f_bf, _ = brute_force_vetorizado(sub)
    _, f_ga, _ = algoritmo_genetico(sub, CONFIGS_PADRAO["A"], 11)
    assert f_ga == pytest.approx(f_bf, rel=1e-6)


def test_ga_parametros_invalidos(modelo):
    with pytest.raises(ValueError):
        algoritmo_genetico(modelo, ConfigGA(populacao=1), 0)
    with pytest.raises(ValueError):
        algoritmo_genetico(modelo, ConfigGA(taxa_mutacao=1.5), 0)


def test_greedy_respeita_orcamento_e_retorno(modelo):
    pos = algoritmo_genetico(modelo, ConfigGA(populacao=30, geracoes=30), 1)[0]
    g = greedy_incentivos(modelo, pos)
    assert g["investimento"].sum() <= modelo.p.orcamento_promocional + 1e-6
    assert (g["investimento"] >= 0).all()
    assert g["ganho_bruto"].sum() >= g["investimento"].sum()  # só investe com retorno marginal > 1


def test_alocar_guloso_otimo_em_caso_simples():
    # ganhos côncavos a_i*sqrt(x); orçamento 100, passo 1
    a = [4.0, 1.0]
    x, _ = alocar_guloso(lambda i, v: a[i] * np.sqrt(v), 2, 100, 1.0, limiar_retorno=0.0)
    assert x.sum() == pytest.approx(100)
    assert x[0] > x[1] * 10  # ótimo analítico: x0/x1 = (a0/a1)^2 = 16


def test_alocar_guloso_entradas_invalidas():
    with pytest.raises(ValueError):
        alocar_guloso(lambda i, v: v, 1, -1, 1.0)


def test_cenario_surpresa_por_parametros(modelo):
    df, cfg = carregar_q1(RAIZ / "data")
    df2, cfg2 = aplicar_cenario_q1(df, cfg, meta_mult=1.1, orcamento_mult=0.75, remover_skus=["Q1-001"])
    assert len(df2) == len(df) - 1
    assert cfg2["meta_caixa_90d"] == pytest.approx(cfg["meta_caixa_90d"] * 1.1)
    with pytest.raises(ValueError):
        aplicar_cenario_q1(df, cfg, remover_skus=["XXX"])


def test_subconjunto_minimo_de_6(modelo):
    with pytest.raises(ValueError):
        selecionar_subconjunto(modelo, 5, 1)
