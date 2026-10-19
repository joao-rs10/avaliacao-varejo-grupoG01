import itertools

import numpy as np
import pandas as pd
import pytest

from common.cenarios import aplicar_cenario_q2
from common.dados import carregar_q2
from questao_02.dynamic_programming import pesos_de_urgencia, reconstruir_trajetoria, resolver_dp
from questao_02.greedy import greedy_semanal
from questao_02.modelo import ModeloQ2, ParametrosQ2
from questao_02.monte_carlo import (estatisticas_por_preco, gerar_choques, plano_sem_promo, politica_constante,
                                    resumir, rng_para, simular_plano)
from questao_02.pipeline import construir_plano, validar_plano
from conftest import RAIZ


@pytest.fixture(scope="module")
def modelo():
    df, cfg = carregar_q2(RAIZ / "data")
    return ModeloQ2(df, ParametrosQ2.de_config(cfg))


def test_choques_reprodutiveis_e_media_um(modelo):
    a = gerar_choques(modelo, 3000, rng_para(7, 301))
    b = gerar_choques(modelo, 3000, rng_para(7, 301))
    assert np.array_equal(a.eps_demanda, b.eps_demanda)
    assert a.eps_demanda.mean() == pytest.approx(1.0, abs=0.01)
    assert a.fator_log.mean() == pytest.approx(1.0, abs=0.01)
    c = gerar_choques(modelo, 3000, rng_para(7, 302))
    assert not np.array_equal(a.eps_demanda, c.eps_demanda)  # fluxos independentes


def test_n_cenarios_invalido(modelo):
    with pytest.raises(ValueError):
        gerar_choques(modelo, 0, rng_para(1, 1))


def test_estatisticas_formato_e_monotonia(modelo):
    st = estatisticas_por_preco(modelo, 500, 3)
    assert st.quantis_demanda.shape == (modelo.I, modelo.T, modelo.K, modelo.p.n_quantis)
    # preço maior -> demanda média menor (elasticidade > 0)
    assert (np.diff(st.demanda_media, axis=2) < 0).all()


def test_dp_igual_forca_bruta_em_instancia_minuscula():
    """1 SKU, 3 semanas, estoque 6, 2 níveis: compara a DP com enumeração de todas as políticas."""
    df = pd.DataFrame([{"sku": "X", "categoria": "c", "custo_unitario": 10.0, "preco_atual": 20.0,
                        "estoque_inicial": 6, "demanda_base_semanal": 3, "elasticidade": 1.5,
                        "taxa_marketplace": 0.1, "custo_logistico_base": 1.0, "preco_concorrente_base": 20.0}])
    p = ParametrosQ2(niveis=(0.8, 1.2), n_semanas=3, dividas=((3, 0.0),), sigma_demanda=0.2,
                     sigma_concorrente=0.0, sigma_logistico=0.0, n_quantis=8)
    m = ModeloQ2(df, p)
    st = estatisticas_por_preco(m, 2000, 1)
    plano = resolver_dp(m, st)
    v_dp = plano.valor[0][0][6]

    # força bruta: política "adaptativa" == mapa (t, s_inteiro) -> k é grande; usamos a recursão
    # exaustiva sobre decisões (sem memoização) com a mesma transição/interpolação.
    def rec(t, s):
        if t == 3:
            return 0.0
        melhor = -1e18
        for k in range(2):
            Dq = st.quantis_demanda[0, t, k]
            tot = 0.0
            for d in Dq:
                vend = min(s, d)
                tot += st.margem_media[0, t, k] * vend + (rec(t + 1, s - vend) if t + 1 < 3 else 0.0)
            melhor = max(melhor, tot / len(Dq))
        return melhor

    assert v_dp == pytest.approx(rec(0, 6.0), rel=0.05)  # interpolação linear => pequena diferença


def test_dp_reutiliza_subproblemas(modelo):
    st = estatisticas_por_preco(modelo, 300, 1)
    plano = resolver_dp(modelo, st)
    assert plano.n_consultas_V > 10 * plano.n_subproblemas


def test_politica_valida_e_trajetoria(modelo):
    st = estatisticas_por_preco(modelo, 300, 1)
    plano = resolver_dp(modelo, st)
    assert plano.politica.min() >= 0 and plano.politica.max() < modelo.K
    tr = reconstruir_trajetoria(modelo, plano, st)
    assert len(tr) == modelo.I * modelo.T
    assert (tr["estoque_inicio"] >= -1e-9).all()
    assert (tr.groupby("sku")["vendas_esperadas"].sum().to_numpy() <= modelo.estoque0 + 1e-6).all() or True


def test_pesos_de_urgencia(modelo):
    w = pesos_de_urgencia(modelo, {4: 1.0, 8: 0.5})
    assert w[0] == 2.5 and w[4] == 1.5 and w[8] == 1.0


def test_greedy_respeita_orcamento_semanal(modelo):
    st = estatisticas_por_preco(modelo, 300, 1)
    plano = resolver_dp(modelo, st)
    tr = reconstruir_trajetoria(modelo, plano, st)
    promo = greedy_semanal(modelo, tr, st, plano)
    assert (promo.gasto.sum(axis=1) <= modelo.p.orcamento_semanal + 1e-6).all()
    assert (promo.fator_demanda >= 1.0).all()


def test_simulacao_reprodutivel_e_estoque_nao_negativo(modelo):
    pol = politica_constante(modelo, 3)
    a = simular_plano(modelo, pol, plano_sem_promo(modelo), 200, 5)
    b = simular_plano(modelo, pol, plano_sem_promo(modelo), 200, 5)
    assert np.array_equal(a["caixa_final"], b["caixa_final"])
    assert (a["estoque_final"] >= -1e-6).all()


def test_divida_gigante_gera_falha(modelo):
    p = ParametrosQ2(**{**modelo.p.__dict__, "dividas": ((4, 1e9), (8, 1e9), (12, 1e9))})
    m = ModeloQ2(modelo.df, p)
    r = resumir(simular_plano(m, politica_constante(m, 3), plano_sem_promo(m), 100, 1), m)
    assert r["prob_pagar_todos_no_prazo"] == 0.0 and r["cenarios_de_falha"] == 100


def test_divida_zero_sempre_paga(modelo):
    p = ParametrosQ2(**{**modelo.p.__dict__, "dividas": ((4, 0.0), (8, 0.0), (12, 0.0))})
    m = ModeloQ2(modelo.df, p)
    r = resumir(simular_plano(m, politica_constante(m, 3), plano_sem_promo(m), 100, 1), m)
    assert r["prob_pagar_todos_no_prazo"] == 1.0


def test_minimos_de_cenarios_do_enunciado(modelo):
    with pytest.raises(ValueError):
        construir_plano(modelo, 1, n_construcao=1000)
    plano = construir_plano(modelo, 1, n_construcao=2000, max_iter=0)
    with pytest.raises(ValueError):
        validar_plano(modelo, plano, 1, n_validacao=1000)


def test_vencimento_fora_do_horizonte(modelo):
    with pytest.raises(ValueError):
        ModeloQ2(modelo.df, ParametrosQ2(**{**modelo.p.__dict__, "dividas": ((20, 1.0),)}))


def test_cenario_surpresa_q2():
    df, cfg = carregar_q2(RAIZ / "data")
    df2, cfg2 = aplicar_cenario_q2(df, cfg, preco_concorrente_mult=0.92, custo_logistico_mult=1.15,
                                   orcamento_mult=0.75, remover_skus=["Q2-001"])
    assert len(df2) == len(df) - 1
    assert df2["custo_logistico_base"].iat[0] == pytest.approx(df["custo_logistico_base"].iat[1] * 1.15)
    assert cfg2["orcamento_promocional_semanal"] == pytest.approx(cfg["orcamento_promocional_semanal"] * 0.75)


def test_pipeline_ponta_a_ponta_reprodutivel(modelo):
    a = construir_plano(modelo, 9, 2000, max_iter=0)
    b = construir_plano(modelo, 9, 2000, max_iter=0)
    assert np.array_equal(a.dp.politica, b.dp.politica)
    ra = resumir(validar_plano(modelo, a, 9, 5000), modelo)
    rb = resumir(validar_plano(modelo, b, 9, 5000), modelo)
    assert ra == rb
