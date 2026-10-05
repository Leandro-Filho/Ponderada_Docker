"""
TESTES DA API

Os testes em test_modelo.py cobrem as features e o artefato. Estes cobrem a
camada HTTP: as rotas respondem? a entrada errada vira 400 e não 500? e o que
acontece quando o artefato não está lá?

Rodam sem Docker, com o test_client do Flask.

    python -m pytest testes/ -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
ARTEFATOS = RAIZ / "artefatos"

# o app.py carrega o artefato no import, então os caminhos vêm antes
os.environ.setdefault("CAMINHO_MODELO", str(ARTEFATOS / "modelo.joblib"))
os.environ.setdefault("CAMINHO_METRICAS", str(ARTEFATOS / "metricas.json"))
os.environ.setdefault("CAMINHO_EXEMPLO", str(ARTEFATOS / "exemplo_requisicao.json"))

sys.path.insert(0, str(RAIZ / "comum"))
sys.path.insert(0, str(RAIZ / "api"))

if not (ARTEFATOS / "modelo.joblib").exists():
    pytest.skip("artefato não existe — rode: python treino/treinar.py",
                allow_module_level=True)

import app as api  # noqa: E402


@pytest.fixture()
def cliente():
    api.app.config["TESTING"] = True
    return api.app.test_client()


@pytest.fixture()
def entrada():
    """A série de exemplo que o treino salvou."""
    return json.loads((ARTEFATOS / "exemplo_requisicao.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------- caminho feliz
def test_indice_lista_as_rotas(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    corpo = r.get_json()
    assert "POST /prever" in corpo["rotas"]
    assert corpo["modelo_carregado"] is True


def test_health_responde_200(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.get_json()["modelo_carregado"] is True


def test_info_traz_o_baseline(cliente):
    """Sem o baseline registrado não dá para afirmar nada sobre o modelo."""
    corpo = cliente.get("/info").get_json()
    assert corpo["metricas_naive"]["rmse_preco_usd"] > 0
    assert isinstance(corpo["bateu_baseline_naive"], bool)
    assert corpo["minimo_historico"] >= 1


def test_exemplo_devolve_predicao(cliente):
    r = cliente.get("/exemplo")
    assert r.status_code == 200
    res = r.get_json()["resultado"]
    assert res["preco_previsto"] > 0
    assert res["direcao"] in ("alta", "baixa", "estável")
    assert "80%" in res["faixas"] and "95%" in res["faixas"]


def test_prever_com_serie_valida(cliente, entrada):
    r = cliente.post("/prever", json=entrada)
    assert r.status_code == 200
    res = r.get_json()
    assert res["dias_de_historico_recebidos"] == len(entrada["precos"])
    assert res["data_prevista"] == "2026-05-24" or res["data_prevista"] is not None


def test_faixa_contem_o_ponto_previsto(cliente, entrada):
    """Invariante: mínimo < previsto < máximo, e a de 95% é mais larga que a de 80%."""
    res = cliente.post("/prever", json=entrada).get_json()
    f80, f95 = res["faixas"]["80%"], res["faixas"]["95%"]
    assert f80["minimo"] < res["preco_previsto"] < f80["maximo"]
    assert f95["minimo"] < f80["minimo"]
    assert f95["maximo"] > f80["maximo"]


def test_predicao_e_deterministica(cliente, entrada):
    """A mesma entrada tem de dar a mesma saída — modelo não é aleatório."""
    a = cliente.post("/prever", json=entrada).get_json()["preco_previsto"]
    b = cliente.post("/prever", json=entrada).get_json()["preco_previsto"]
    assert a == b


def test_variacao_bate_com_os_precos(cliente, entrada):
    """A variação percentual tem de ser coerente com atual e previsto."""
    res = cliente.post("/prever", json=entrada).get_json()
    esperada = (res["preco_previsto"] / res["preco_atual"] - 1) * 100
    assert abs(res["variacao_prevista_pct"] - esperada) < 0.01


# ------------------------------------------------- entrada errada -> 400
@pytest.mark.parametrize("corpo, pedaco_da_mensagem", [
    ({}, "obrigatório"),
    ({"precos": [50000.0] * 70}, "juros"),
    ({"juros": [4.0] * 70}, "precos"),
    ({"precos": [50000.0] * 70, "juros": [4.0] * 30}, "tamanhos diferentes"),
    ({"precos": [50000.0] * 10, "juros": [4.0] * 10}, "histórico insuficiente"),
    ({"precos": ["abc"] * 70, "juros": [4.0] * 70}, "não é número"),
    ({"precos": [-1.0] * 70, "juros": [4.0] * 70}, "positivo"),
])
def test_entrada_invalida_devolve_400(cliente, corpo, pedaco_da_mensagem):
    r = cliente.post("/prever", json=corpo)
    assert r.status_code == 400, f"esperava 400, veio {r.status_code}"
    assert pedaco_da_mensagem in r.get_json()["erro"]


def test_post_sem_json_devolve_400(cliente):
    r = cliente.post("/prever")
    assert r.status_code == 400
    assert "JSON" in r.get_json()["erro"]


def test_rota_inexistente_devolve_404(cliente):
    assert cliente.get("/nao-existe").status_code == 404


def test_metodo_errado_devolve_405(cliente):
    assert cliente.get("/prever").status_code == 405


def test_nenhuma_entrada_ruim_causa_500(cliente):
    """Varredura: nada que o cliente mande pode derrubar o servidor."""
    ruins = [
        {"precos": None, "juros": None},
        {"precos": [], "juros": []},
        {"precos": "texto", "juros": "texto"},
        {"precos": [{"a": 1}] * 70, "juros": [4.0] * 70},
        {"precos": [50000.0] * 70, "juros": [4.0] * 70, "data_final": "data-invalida"},
        {"precos": [float("inf")] * 70, "juros": [4.0] * 70},
    ]
    for corpo in ruins:
        r = cliente.post("/prever", json=corpo)
        assert r.status_code < 500, f"{corpo} causou HTTP {r.status_code}"


# ------------------------------------------------ sem artefato -> 503
def test_sem_artefato_health_continua_200_e_resto_da_503(cliente):
    """Se o artefato sumir, o /health segue 200 (senão o container reinicia em
    loop) mas as rotas que dependem do modelo devolvem 503 com o motivo."""
    original = api.CAMINHO_MODELO
    try:
        api.CAMINHO_MODELO = Path("/caminho/que/nao/existe.joblib")
        api.carregar_artefato()

        r = cliente.get("/health")
        assert r.status_code == 200
        assert r.get_json()["modelo_carregado"] is False

        for rota in ("/info", "/exemplo"):
            r = cliente.get(rota)
            assert r.status_code == 503
            assert "indisponível" in r.get_json()["erro"]

        r = cliente.post("/prever", json={"precos": [1.0] * 70, "juros": [4.0] * 70})
        assert r.status_code == 503
    finally:
        api.CAMINHO_MODELO = original
        api.carregar_artefato()          # devolve o estado para os outros testes
        assert api.pronto()


def test_recarregar_funciona(cliente):
    r = cliente.get("/recarregar")
    assert r.status_code == 200
    assert r.get_json()["recarregado"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))