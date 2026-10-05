"""
TESTES DO MODELO

Roda com pytest ou direto:
    pytest testes/ -v
    python testes/test_modelo.py

O teste que mais importa é o `test_features_nao_olham_o_futuro`. Ele é a
prova mecânica de que nenhuma feature usa informação de amanhã — e vazamento
de dados é o erro que mais derruba projeto de série temporal, justamente
porque não dá erro: o modelo vai bem no teste e falha em produção.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import joblib
except ModuleNotFoundError:                                    # pragma: no cover
    pytest.exit(
        "\nERRO: joblib não encontrado.\n"
        "Quase sempre isto é o pytest rodando num Python diferente do seu venv.\n"
        "Confira a linha 'platform ...' no topo da saída do pytest: ela mostra\n"
        "qual interpretador está em uso.\n\n"
        "Correção:\n"
        "    pip install -r requirements-dev.txt\n"
        "    python -m pytest testes/ -v        <- com 'python -m', não 'pytest' solto\n",
        returncode=1)

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "comum"))
import features as ft  # noqa: E402

DATASET = RAIZ / "dados" / "dataset.csv"
ARTEFATO = RAIZ / "artefatos" / "modelo.joblib"


@pytest.fixture(scope="module")
def dados() -> pd.DataFrame:
    if not DATASET.exists():
        pytest.skip(f"{DATASET} não existe — rode: python dados/preparar.py")
    return pd.read_csv(DATASET, parse_dates=["data"]).sort_values("data").reset_index(drop=True)


# ---------------------------------------------------------------- features
def test_features_nao_olham_o_futuro(dados):
    """O teste central contra vazamento.

    Calcula as features com a série INTEIRA e depois com a série CORTADA no
    dia t. Se alguma feature usasse informação posterior a t, os dois valores
    seriam diferentes. Têm de ser idênticos.
    """
    completo = ft.calcular(dados["preco_btc_usd"], dados["juros_pct"], dados["data"])

    for t in (200, 600, 1000, len(dados) - 1):
        cortado = ft.calcular(
            dados["preco_btc_usd"].iloc[: t + 1],
            dados["juros_pct"].iloc[: t + 1],
            dados["data"].iloc[: t + 1],
        )
        a = completo.iloc[t].to_numpy(dtype=float)
        b = cortado.iloc[t].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), (
            f"no dia {t} ({dados['data'].iloc[t].date()}) as features mudaram quando "
            f"a série foi cortada — alguma feature está olhando para o futuro"
        )


def test_features_completas_apos_aquecimento(dados):
    """Depois do período de aquecimento, nenhuma feature pode ficar nula."""
    f = ft.calcular(dados["preco_btc_usd"], dados["juros_pct"], dados["data"])
    depois = f.iloc[ft.MINIMO_HISTORICO:]
    nulas = depois.columns[depois.isna().any()].tolist()
    assert not nulas, f"features com nulo após o aquecimento: {nulas}"


def test_ordem_das_features_e_contrato(dados):
    f = ft.calcular(dados["preco_btc_usd"], dados["juros_pct"], dados["data"])
    assert list(f.columns) == ft.FEATURES, (
        "a ordem das colunas mudou — isso quebra o contrato com o artefato treinado"
    )


# ------------------------------------------------------- validação de entrada
def test_recusa_historico_curto():
    n = ft.MINIMO_HISTORICO - 1
    with pytest.raises(ValueError, match="histórico insuficiente"):
        ft.ultima_linha([100.0] * n, [4.0] * n)


def test_recusa_tamanhos_diferentes():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        ft.ultima_linha([100.0] * 80, [4.0] * 70)


def test_recusa_preco_nao_positivo():
    n = ft.MINIMO_HISTORICO + 5
    with pytest.raises(ValueError, match="positivo"):
        ft.ultima_linha([-1.0] * n, [4.0] * n)


def test_recusa_valor_nulo():
    n = ft.MINIMO_HISTORICO + 5
    precos = [100.0] * n
    precos[10] = float("nan")
    with pytest.raises(ValueError, match="nulo|não numérico"):
        ft.ultima_linha(precos, [4.0] * n)


def test_aceita_historico_minimo(dados):
    """Com exatamente MINIMO_HISTORICO dias tem de funcionar, sem nulo."""
    n = ft.MINIMO_HISTORICO
    linha = ft.ultima_linha(
        dados["preco_btc_usd"].iloc[:n], dados["juros_pct"].iloc[:n], dados["data"].iloc[:n])
    assert linha.shape == (1, len(ft.FEATURES))
    assert not linha.isna().any().any()


# ------------------------------------------------------------------- alvo
def test_alvo_e_o_dia_seguinte(dados):
    """preco_hoje * exp(alvo) tem de reconstruir exatamente o preço de amanhã."""
    p = dados["preco_btc_usd"]
    alvo = np.log(p.shift(-1) / p)
    reconstruido = (p * np.exp(alvo)).iloc[:-1]
    assert np.allclose(reconstruido, p.shift(-1).iloc[:-1], rtol=1e-9)


# --------------------------------------------------------------- artefato
@pytest.fixture(scope="module")
def artefato():
    if not ARTEFATO.exists():
        pytest.skip(f"{ARTEFATO} não existe — rode: python treino/treinar.py")
    return joblib.load(ARTEFATO)


def test_artefato_tem_as_chaves_que_a_api_usa(artefato):
    for chave in ("pipeline", "features", "minimo_historico", "nome_modelo",
                  "calibracao_faixa", "metricas_teste", "treinado_em"):
        assert chave in artefato, f"o artefato não tem a chave '{chave}'"


def test_artefato_casa_com_o_features_py(artefato):
    """Se o features.py mudar e o modelo não for re-treinado, isso pega."""
    assert artefato["features"] == ft.FEATURES, (
        "a lista de features do artefato difere da do comum/features.py — "
        "re-treine o modelo (training/serving skew)"
    )
    assert artefato["minimo_historico"] == ft.MINIMO_HISTORICO


def test_artefato_preve_valor_plausivel(artefato, dados):
    """Predição de ponta a ponta: tem de sair um retorno pequeno e finito."""
    n = ft.MINIMO_HISTORICO + 9
    X = ft.ultima_linha(
        dados["preco_btc_usd"].tail(n), dados["juros_pct"].tail(n), dados["data"].tail(n))
    retorno = float(artefato["pipeline"].predict(X)[0])
    assert np.isfinite(retorno), "a predição não é um número finito"
    assert abs(retorno) < 0.5, (
        f"retorno diário previsto de {retorno*100:.1f}% é absurdo para BTC em 1 dia"
    )


def test_faixa_esta_calibrada(artefato):
    """A faixa de 80% tem de cobrir perto de 80% no teste. Tolerância de 7 pontos."""
    for nivel in (80, 95):
        medido = artefato["calibracao_faixa"][f"nivel_{nivel}"]["cobertura_empirica_pct"]
        assert abs(medido - nivel) <= 7, (
            f"faixa de {nivel}% cobriu {medido:.1f}% no teste — descalibrada"
        )


def test_baseline_naive_foi_registrado(artefato):
    """Sem o baseline não dá para afirmar nada sobre o modelo."""
    assert "metricas_naive" in artefato
    assert artefato["metricas_naive"]["rmse_preco_usd"] > 0
    assert isinstance(artefato["bateu_baseline_naive"], bool)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))