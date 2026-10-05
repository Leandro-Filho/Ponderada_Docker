"""
API DE INFERÊNCIA — carrega o artefato treinado e serve predições.

Como o artefato chega aqui: o container de treino grava /artefatos/modelo.joblib
num VOLUME NOMEADO; este container monta o mesmo volume em SOMENTE LEITURA.
Os dois nunca se falam diretamente — o contrato entre eles é o arquivo.

Rotas:
  GET  /                 a página de demonstração (a aplicação cliente)
  GET  /health           o serviço está vivo? (usado pelo HEALTHCHECK do Docker)
  GET  /info             qual modelo está carregado, quando foi treinado
  GET  /modelo           relatório completo de métricas e calibração
  GET  /exemplo          predição com os últimos dias reais — demo de um clique
  POST /prever           {"precos": [...], "juros": [...]} -> predição
  GET  /recarregar       relê o artefato do disco sem reiniciar o container
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
from flask import Flask, jsonify, request, send_from_directory
from scipy.stats import norm

# o MESMO módulo de features usado no treino
sys.path.insert(0, "/app/comum")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comum"))
import features as ft  # noqa: E402

AQUI = Path(__file__).resolve().parent
CAMINHO_MODELO = Path(os.environ.get("CAMINHO_MODELO", "/artefatos/modelo.joblib"))
CAMINHO_METRICAS = Path(os.environ.get("CAMINHO_METRICAS", "/artefatos/metricas.json"))
CAMINHO_EXEMPLO = Path(os.environ.get("CAMINHO_EXEMPLO", "/artefatos/exemplo_requisicao.json"))
NIVEIS = (80, 95)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | API | %(levelname)-7s | %(message)s",
)
reg = logging.getLogger("api")

app = Flask(__name__, static_folder=None)
HOSTNAME = socket.gethostname()

# ----------------------------------------------------------- carregar artefato
# Carregado UMA VEZ, na subida do processo — não a cada requisição.
# Se o arquivo não existir, o serviço sobe assim mesmo e responde 503. Se ele
# morresse, o container entraria em loop de reinício e a mensagem de erro se
# perderia; subindo, o /health continua respondendo e o motivo fica legível.
ARTEFATO: dict | None = None
ERRO_CARGA: str | None = None


def carregar_modelo() -> None:
    global ARTEFATO, ERRO_CARGA
    try:
        ARTEFATO = joblib.load(CAMINHO_MODELO)
        ERRO_CARGA = None
        reg.info("artefato carregado: %s (v%s), treinado em %s",
                 ARTEFATO["nome_modelo"], ARTEFATO["versao_artefato"], ARTEFATO["treinado_em"])
        # guarda contra training/serving skew: se o features.py mudou e o
        # modelo não foi re-treinado, o serviço recusa em vez de prever errado
        if ARTEFATO["features"] != ft.FEATURES:
            ERRO_CARGA = ("o artefato foi treinado com outra lista de features — "
                          "re-treine o modelo (comum/features.py mudou)")
            reg.error(ERRO_CARGA)
    except FileNotFoundError:
        ERRO_CARGA = (f"artefato não encontrado em {CAMINHO_MODELO}. "
                      "Rode o treino antes: docker compose run --rm treino")
        reg.warning(ERRO_CARGA)
    except Exception as exc:                                        # noqa: BLE001
        ERRO_CARGA = f"falha ao carregar o artefato: {exc}"
        reg.error(ERRO_CARGA)


carregar_modelo()


def exige_modelo():
    """Devolve uma resposta 503 se o modelo não estiver pronto, ou None."""
    if ARTEFATO is None or ERRO_CARGA:
        return jsonify({"erro": "modelo indisponível", "detalhe": ERRO_CARGA,
                        "servido_por": HOSTNAME}), 503
    return None


# --------------------------------------------------------------- aplicação cliente
@app.get("/")
def pagina():
    """A própria API serve a página. Mesma origem = zero CORS, um container a menos."""
    return send_from_directory(AQUI, "pagina.html")


# --------------------------------------------------------------------- rotas
@app.get("/health")
def health():
    """Liveness. Responde 200 mesmo sem modelo; quem diz a verdade é o campo
    modelo_carregado."""
    return jsonify({
        "status": "ok",
        "servico": "api-inferencia",
        "modelo_carregado": ARTEFATO is not None and not ERRO_CARGA,
        "servido_por": HOSTNAME,
    }), 200


@app.get("/info")
def info():
    if (r := exige_modelo()):
        return r
    a = ARTEFATO
    return jsonify({
        "modelo": a["nome_modelo"],
        "versao_artefato": a["versao_artefato"],
        "tipo_predicao": a["tipo_predicao"],
        "treinado_em": a["treinado_em"],
        "periodo_treino": a["periodo_treino"],
        "ultima_data_conhecida": a["ultima_data_conhecida"],
        "ultimo_preco_conhecido": a["ultimo_preco_conhecido"],
        "ultimo_juros_conhecido": a["ultimo_juros_conhecido"],
        "features": a["features"],
        "minimo_historico": a["minimo_historico"],
        "bateu_baseline_naive": a["bateu_baseline_naive"],
        "versao_sklearn": a["versao_sklearn"],
        "servido_por": HOSTNAME,
    }), 200


@app.get("/modelo")
def modelo():
    if (r := exige_modelo()):
        return r
    if CAMINHO_METRICAS.exists():
        return jsonify(json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))), 200
    return jsonify({"metricas_teste": ARTEFATO["metricas_teste"],
                    "metricas_naive": ARTEFATO["metricas_naive"],
                    "calibracao_faixa": ARTEFATO["calibracao_faixa"]}), 200


def _prever(precos: list[float], juros: list[float], data_final: str | None) -> dict:
    """Coração da inferência. Usa comum/features.py — o MESMO código do treino."""
    datas = None
    if data_final:
        fim = date.fromisoformat(data_final)
        datas = [fim - timedelta(days=i) for i in range(len(precos) - 1, -1, -1)]

    X = ft.ultima_linha(precos, juros, datas)            # valida e calcula
    retorno = float(ARTEFATO["pipeline"].predict(X)[0])  # retorno log previsto

    preco_hoje = float(precos[-1])
    preco_prev = preco_hoje * float(np.exp(retorno))
    sigma = float(X["volatilidade_ewma"].iloc[0])

    faixas = {}
    for nivel in NIVEIS:
        z = float(norm.ppf(0.5 + nivel / 200))
        faixas[f"{nivel}%"] = {
            "minimo": round(preco_hoje * float(np.exp(retorno - z * sigma)), 2),
            "maximo": round(preco_hoje * float(np.exp(retorno + z * sigma)), 2),
            "cobertura_medida_no_teste_pct": ARTEFATO["calibracao_faixa"]
                .get(f"nivel_{nivel}", {}).get("cobertura_empirica_pct"),
        }

    return {
        "data_prevista": str(date.fromisoformat(data_final) + timedelta(days=1))
                         if data_final else None,
        "preco_atual": round(preco_hoje, 2),
        "preco_previsto": round(preco_prev, 2),
        "variacao_prevista_pct": round((float(np.exp(retorno)) - 1) * 100, 4),
        "retorno_log_previsto": round(retorno, 6),
        "direcao": "alta" if retorno > 0 else ("baixa" if retorno < 0 else "estável"),
        "faixas": faixas,
        "volatilidade_diaria_estimada_pct": round(sigma * 100, 3),
        "juros_atual_pct": round(float(juros[-1]), 4),
        "regime_juros": ("subindo" if X["regime_subindo"].iloc[0] == 1 else
                         "cortando" if X["regime_cortando"].iloc[0] == 1 else "parado"),
        "dias_de_historico_recebidos": len(precos),
        "modelo": ARTEFATO["nome_modelo"],
        "versao_artefato": ARTEFATO["versao_artefato"],
        "servido_por": HOSTNAME,
        "aviso": "Predição experimental, para fins acadêmicos. "
                 "Não é recomendação de investimento.",
    }


@app.post("/prever")
def prever():
    if (r := exige_modelo()):
        return r

    corpo = request.get_json(silent=True)
    if not isinstance(corpo, dict):
        return jsonify({"erro": "envie um JSON com Content-Type: application/json"}), 400

    faltando = [c for c in ("precos", "juros")
                if not isinstance(corpo.get(c), list) or not corpo.get(c)]
    if faltando:
        return jsonify({
            "erro": f"campo(s) obrigatório(s) ausente(s) ou vazio(s): {faltando}",
            "formato": {"precos": "[fechamentos diários do BTC, do mais antigo ao mais recente]",
                        "juros": "[taxa de juros em %, alinhada dia a dia com os preços]",
                        "data_final": "AAAA-MM-DD (opcional)"},
            "minimo_necessario": ARTEFATO["minimo_historico"],
        }), 400

    try:
        precos = [float(v) for v in corpo["precos"]]
        juros = [float(v) for v in corpo["juros"]]
    except (TypeError, ValueError):
        return jsonify({"erro": "há valor que não é número em 'precos' ou 'juros'"}), 400

    try:
        resultado = _prever(precos, juros, corpo.get("data_final"))
    except ValueError as exc:
        # erro do chamador: histórico curto, tamanhos diferentes, preço negativo
        return jsonify({"erro": str(exc),
                        "minimo_necessario": ARTEFATO["minimo_historico"]}), 400
    except Exception as exc:                                        # noqa: BLE001
        reg.exception("falha inesperada na predição")
        return jsonify({"erro": "falha interna na predição", "detalhe": str(exc)}), 500

    reg.info("predição: %.2f -> %.2f (%+.2f%%) | juros %.2f%% (%s)",
             resultado["preco_atual"], resultado["preco_previsto"],
             resultado["variacao_prevista_pct"], resultado["juros_atual_pct"],
             resultado["regime_juros"])
    return jsonify(resultado), 200


@app.get("/exemplo")
def exemplo():
    """Demo de um clique: roda a predição com os últimos dias reais que o
    treino deixou salvos. É o que se mostra na apresentação."""
    if (r := exige_modelo()):
        return r
    if not CAMINHO_EXEMPLO.exists():
        return jsonify({"erro": f"exemplo não encontrado em {CAMINHO_EXEMPLO}"}), 404
    ex = json.loads(CAMINHO_EXEMPLO.read_text(encoding="utf-8"))
    return jsonify({
        "entrada": {"dias": len(ex["precos"]),
                    "data_final": ex.get("data_final"),
                    "precos": ex["precos"],
                    "juros": ex["juros"]},
        "resultado": _prever(ex["precos"], ex["juros"], ex.get("data_final")),
    }), 200


@app.get("/recarregar")
def recarregar():
    """Relê o artefato do disco sem reiniciar o container. É o que permite
    re-treinar com o serviço no ar."""
    carregar_modelo()
    pronto = ARTEFATO is not None and not ERRO_CARGA
    return jsonify({"recarregado": pronto, "detalhe": ERRO_CARGA,
                    "treinado_em": ARTEFATO.get("treinado_em") if pronto else None,
                    "servido_por": HOSTNAME}), (200 if pronto else 503)


if __name__ == "__main__":
    # Só para rodar fora do Docker. Em container quem serve é o gunicorn.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
