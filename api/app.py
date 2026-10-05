"""
API — a caixa "container 2" do diagrama de componentes.

    volume artefatos ──(joblib.load, :ro)──→ app.py ──HTTP──→ cliente
                                               ↑
                                        comum/features.py

As funções abaixo são, na ordem, as mensagens do DIAGRAMA DE SEQUÊNCIA:

    POST /prever
      1. validar_corpo()        "valida o corpo"      -> 400 se inválido
      2. calcular_features()    A->>F  e  F-->>A
      3. predizer_retorno()     A->>M  e  M-->>A
      4. montar_resposta()      "preço = preço_hoje × e^retorno; faixa = ± z × σ"
      5. devolve 200

Rotas:
    GET  /            índice: lista as rotas (útil no curl)
    GET  /health      o serviço está vivo?  (o enunciado pede isto)
    GET  /info        qual modelo está carregado, métricas e calibração
    POST /prever      {"precos": [...], "juros": [...]}  -> predição
    GET  /exemplo     predição com os últimos dias reais (demo de um clique)
    GET  /recarregar  relê o artefato sem reiniciar o container
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import norm

sys.path.insert(0, "/app/comum")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comum"))
import features as ft  # noqa: E402  (o MESMO módulo que o treino usa)

CAMINHO_MODELO = Path(os.environ.get("CAMINHO_MODELO", "/artefatos/modelo.joblib"))
CAMINHO_METRICAS = Path(os.environ.get("CAMINHO_METRICAS", "/artefatos/metricas.json"))
CAMINHO_EXEMPLO = Path(os.environ.get("CAMINHO_EXEMPLO", "/artefatos/exemplo_requisicao.json"))
NIVEIS = (80, 95)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s | API | %(levelname)-7s | %(message)s")
reg = logging.getLogger("api")

app = Flask(__name__, static_folder=None)
HOSTNAME = socket.gethostname()

MODELO: dict | None = None     # o artefato carregado em memória
ERRO: str | None = None


# ====================================================================
# O ARTEFATO — carregado UMA VEZ, na subida do processo
# ====================================================================
def carregar_artefato() -> None:
    """joblib.load do volume compartilhado. É por aqui que o modelo treinado
    entra neste container."""
    global MODELO, ERRO
    try:
        MODELO = joblib.load(CAMINHO_MODELO)
        ERRO = None
        reg.info("artefato carregado: %s, treinado em %s",
                 MODELO["nome_modelo"], MODELO["treinado_em"])
        # guarda contra training/serving skew
        if MODELO["features"] != ft.FEATURES:
            ERRO = ("o artefato foi treinado com outra lista de features — "
                    "re-treine o modelo (comum/features.py mudou)")
            reg.error(ERRO)
    except FileNotFoundError:
        ERRO = (f"artefato não encontrado em {CAMINHO_MODELO}. "
                "Rode o treino: docker compose run --rm treino")
        reg.warning(ERRO)
    except Exception as exc:                                        # noqa: BLE001
        ERRO = f"falha ao carregar o artefato: {exc}"
        reg.error(ERRO)


carregar_artefato()


def pronto() -> bool:
    return MODELO is not None and ERRO is None


# ====================================================================
# OS 4 PASSOS DO DIAGRAMA DE SEQUÊNCIA
# ====================================================================
def validar_corpo(corpo) -> tuple[list[float], list[float], str | None]:
    """Passo 1 — "valida o corpo". Levanta ValueError, que vira 400."""
    if not isinstance(corpo, dict):
        raise ValueError("envie um JSON com Content-Type: application/json")

    for campo in ("precos", "juros"):
        if not isinstance(corpo.get(campo), list) or not corpo[campo]:
            raise ValueError(f"campo '{campo}' é obrigatório e precisa ser uma lista não vazia")
    try:
        precos = [float(v) for v in corpo["precos"]]
        juros = [float(v) for v in corpo["juros"]]
    except (TypeError, ValueError):
        raise ValueError("há valor que não é número em 'precos' ou 'juros'")

    return precos, juros, corpo.get("data_final")


def calcular_features(precos, juros, data_final):
    """Passo 2 — A->>F e F-->>A. O MESMO features.py que o treino usou.
    Também valida tamanho e sinal, levantando ValueError (-> 400)."""
    datas = None
    if data_final:
        fim = date.fromisoformat(data_final)
        datas = [fim - timedelta(days=i) for i in range(len(precos) - 1, -1, -1)]
    return ft.ultima_linha(precos, juros, datas)


def predizer_retorno(X) -> float:
    """Passo 3 — A->>M e M-->>A. O modelo devolve o retorno log de amanhã."""
    return float(MODELO["pipeline"].predict(X)[0])


def montar_resposta(X, retorno, precos, juros, data_final) -> dict:
    """Passo 4 — converte retorno em preço e calcula a faixa de incerteza."""
    preco_hoje = float(precos[-1])
    sigma = float(X["volatilidade_ewma"].iloc[0])

    faixas = {}
    for nivel in NIVEIS:
        z = float(norm.ppf(0.5 + nivel / 200))
        faixas[f"{nivel}%"] = {
            "minimo": round(preco_hoje * float(np.exp(retorno - z * sigma)), 2),
            "maximo": round(preco_hoje * float(np.exp(retorno + z * sigma)), 2),
            "cobertura_medida_no_teste_pct":
                MODELO["calibracao_faixa"].get(f"nivel_{nivel}", {}).get("cobertura_empirica_pct"),
        }

    return {
        "data_prevista": str(date.fromisoformat(data_final) + timedelta(days=1))
                         if data_final else None,
        "preco_atual": round(preco_hoje, 2),
        "preco_previsto": round(preco_hoje * float(np.exp(retorno)), 2),
        "variacao_prevista_pct": round((float(np.exp(retorno)) - 1) * 100, 4),
        "retorno_log_previsto": round(retorno, 6),
        "direcao": "alta" if retorno > 0 else ("baixa" if retorno < 0 else "estável"),
        "faixas": faixas,
        "volatilidade_diaria_estimada_pct": round(sigma * 100, 3),
        "juros_atual_pct": round(float(juros[-1]), 4),
        "regime_juros": ("subindo" if X["regime_subindo"].iloc[0] == 1 else
                         "cortando" if X["regime_cortando"].iloc[0] == 1 else "parado"),
        "dias_de_historico_recebidos": len(precos),
        "modelo": MODELO["nome_modelo"],
        "servido_por": HOSTNAME,
        "aviso": "Predição experimental, para fins acadêmicos. "
                 "Não é recomendação de investimento.",
    }


def prever(precos, juros, data_final) -> dict:
    """Os 4 passos em sequência — é esta função que o diagrama desenha."""
    X = calcular_features(precos, juros, data_final)
    retorno = predizer_retorno(X)
    return montar_resposta(X, retorno, precos, juros, data_final)


# ====================================================================
# ROTAS
# ====================================================================
@app.get("/")
def indice():
    """Índice das rotas. Um `curl localhost:8000` já mostra o que dá para chamar.
    Não há front-end: a aplicação cliente é o curl (ou o Postman)."""
    return jsonify({
        "servico": "api-inferencia-btc",
        "modelo_carregado": pronto(),
        "rotas": {
            "GET /health": "o serviço está vivo",
            "GET /info": "qual modelo, métricas e calibração",
            "POST /prever": "{'precos': [...], 'juros': [...], 'data_final': 'AAAA-MM-DD'}",
            "GET /exemplo": "predição com os últimos dias reais",
            "GET /recarregar": "relê o artefato sem reiniciar",
        },
        "minimo_historico": MODELO["minimo_historico"] if pronto() else None,
    }), 200


@app.get("/health")
def health():
    """Responde 200 mesmo sem modelo, para o container não reiniciar em loop.
    Quem diz a verdade é o campo modelo_carregado."""
    return jsonify({"status": "ok", "modelo_carregado": pronto(),
                    "servido_por": HOSTNAME}), 200


@app.get("/info")
def info():
    if not pronto():
        return jsonify({"erro": "modelo indisponível", "detalhe": ERRO}), 503
    resposta = {
        "modelo": MODELO["nome_modelo"],
        "treinado_em": MODELO["treinado_em"],
        "periodo_treino": MODELO["periodo_treino"],
        "ultima_data_conhecida": MODELO["ultima_data_conhecida"],
        "features": MODELO["features"],
        "minimo_historico": MODELO["minimo_historico"],
        "metricas_teste": MODELO["metricas_teste"],
        "metricas_naive": MODELO["metricas_naive"],
        "bateu_baseline_naive": MODELO["bateu_baseline_naive"],
        "calibracao_faixa": MODELO["calibracao_faixa"],
        "servido_por": HOSTNAME,
    }
    if CAMINHO_METRICAS.exists():       # o relatório completo, se estiver no volume
        resposta["relatorio_treino"] = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return jsonify(resposta), 200


@app.post("/prever")
def rota_prever():
    if not pronto():
        return jsonify({"erro": "modelo indisponível", "detalhe": ERRO}), 503
    try:
        precos, juros, data_final = validar_corpo(request.get_json(silent=True))
        resultado = prever(precos, juros, data_final)
    except ValueError as exc:
        # erro do chamador: corpo errado, histórico curto, preço negativo
        return jsonify({"erro": str(exc),
                        "minimo_necessario": MODELO["minimo_historico"]}), 400
    except Exception as exc:                                        # noqa: BLE001
        reg.exception("falha inesperada na predição")
        return jsonify({"erro": "falha interna", "detalhe": str(exc)}), 500

    reg.info("predição: %.2f -> %.2f (%+.2f%%)", resultado["preco_atual"],
             resultado["preco_previsto"], resultado["variacao_prevista_pct"])
    return jsonify(resultado), 200


@app.get("/exemplo")
def exemplo():
    """Demo de um clique, com os últimos dias reais que o treino salvou."""
    if not pronto():
        return jsonify({"erro": "modelo indisponível", "detalhe": ERRO}), 503
    if not CAMINHO_EXEMPLO.exists():
        return jsonify({"erro": f"exemplo não encontrado em {CAMINHO_EXEMPLO}"}), 404
    ex = json.loads(CAMINHO_EXEMPLO.read_text(encoding="utf-8"))
    return jsonify({"entrada": ex,
                    "resultado": prever(ex["precos"], ex["juros"], ex.get("data_final"))}), 200


@app.get("/recarregar")
def recarregar():
    """Relê o artefato do disco — permite re-treinar com a API no ar."""
    carregar_artefato()
    return jsonify({"recarregado": pronto(), "detalhe": ERRO,
                    "treinado_em": MODELO["treinado_em"] if pronto() else None}), \
        (200 if pronto() else 503)


if __name__ == "__main__":
    # Só fora do Docker. Em container quem serve é o gunicorn.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))