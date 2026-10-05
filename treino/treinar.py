"""
TREINAMENTO — estima o preço de fechamento do BTC no dia seguinte,
a partir do histórico do próprio BTC e do juros americano.

Roda dentro do container de treino: lê /dados/dataset.csv e grava o artefato
em /artefatos, o volume compartilhado com a API. É assim que o modelo chega
na inferência.

AS QUATRO DECISÕES, e o porquê de cada uma:

1. O alvo é o RETORNO logarítmico de amanhã, não o preço.
   Prevendo o preço direto, o modelo só precisa copiar o valor de ontem: o R²
   passa de 0,99 e parece ótimo sem ter aprendido nada. Prevendo a variação,
   ele é obrigado a dizer algo sobre o movimento, e o erro fica comparável ao
   do baseline.

2. Split CRONOLÓGICO, sem embaralhar.
   shuffle=True em série temporal é vazamento: o modelo treinaria vendo dias
   posteriores aos que vai prever.

3. O baseline NAIVE é treinado e medido junto.
   "Amanhã fecha igual a hoje". Sem ele não dá para afirmar que o modelo
   aprendeu alguma coisa.

4. Além do ponto, estima a FAIXA de incerteza pela volatilidade EWMA, e mede
   a calibração dessa faixa no teste. O ponto é quase imprevisível; a
   incerteza não é.

Uso:
    python treinar.py
    DADOS=... ARTEFATOS=... PROP_TESTE=0.2 python treinar.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# o módulo compartilhado com a API
sys.path.insert(0, "/app/comum")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comum"))
import features as ft  # noqa: E402

DADOS      = Path(os.environ.get("DADOS", "/dados/dataset.csv"))
ARTEFATOS  = Path(os.environ.get("ARTEFATOS", "/artefatos"))
PROP_TESTE = float(os.environ.get("PROP_TESTE", "0.2"))
SEMENTE    = int(os.environ.get("SEMENTE", "42"))
NIVEIS     = (80, 95)


def log(m: str) -> None:
    print(f"[treino] {m}", flush=True)


def avaliar(preco_hoje, ret_real, ret_prev) -> dict:
    """Mede nas duas escalas: no retorno (o que o modelo prevê) e no preço
    reconstruído (o que o usuário final enxerga)."""
    real = preco_hoje * np.exp(ret_real)
    prev = preco_hoje * np.exp(ret_prev)
    return {
        "rmse_preco_usd": float(np.sqrt(mean_squared_error(real, prev))),
        "mae_preco_usd": float(mean_absolute_error(real, prev)),
        "mape_preco_pct": float(np.mean(np.abs((real - prev) / real)) * 100),
        "rmse_retorno_pp": float(np.sqrt(mean_squared_error(ret_real, ret_prev)) * 100),
        "acuracia_direcao_pct": float((np.sign(ret_prev) == np.sign(ret_real)).mean() * 100),
    }


def main() -> None:
    log(f"python {platform.python_version()} | scikit-learn {sklearn.__version__} | semente {SEMENTE}")

    # ------------------------------------------------------------- carregar
    if not DADOS.exists():
        raise SystemExit(
            f"ERRO: não encontrei {DADOS}.\n"
            "Rode antes:  python dados/preparar.py\n"
            "e confira o volume montado em /dados no docker-compose.yml.")
    d = pd.read_csv(DADOS, parse_dates=["data"]).sort_values("data").reset_index(drop=True)
    log(f"dataset: {len(d)} linhas | {d['data'].min().date()} a {d['data'].max().date()}")
    log(f"juros observado em {int(d['juros_observado'].sum())} dias, "
        f"preenchido em {int((1-d['juros_observado']).sum())} "
        f"({(1-d['juros_observado'].mean())*100:.1f}%)")

    # ------------------------ features pelo MESMO código que a API vai usar
    X = ft.calcular(d["preco_btc_usd"], d["juros_pct"], d["data"])
    y = np.log(d["preco_btc_usd"].shift(-1) / d["preco_btc_usd"])   # alvo: retorno de amanhã
    ok = X.notna().all(axis=1) & y.notna()

    X = X[ok].reset_index(drop=True)
    y = y[ok].reset_index(drop=True)
    preco = d.loc[ok, "preco_btc_usd"].reset_index(drop=True)
    datas = d.loc[ok, "data"].reset_index(drop=True)
    log(f"descartei {int((~ok).sum())} linha(s): aquecimento das janelas + último dia sem amanhã")
    log(f"{len(X)} exemplos, {len(ft.FEATURES)} features (de comum/features.py)")

    # -------------------------------------------------- split cronológico
    corte = int(len(X) * (1 - PROP_TESTE))
    log(f"treino: {corte} dias ({datas.iloc[0].date()} a {datas.iloc[corte-1].date()})")
    log(f"teste : {len(X)-corte} dias ({datas.iloc[corte].date()} a {datas.iloc[-1].date()})")
    log("split por DATA, sem shuffle — embaralhar série temporal é vazamento")

    X_tr, X_te = X.iloc[:corte], X.iloc[corte:]
    y_tr, y_te = y.iloc[:corte], y.iloc[corte:]
    preco_te = preco.iloc[corte:].to_numpy()

    # ---------------------------------------------------------- candidatos
    candidatos = {
        # baseline: prevê retorno ZERO => "amanhã fecha igual a hoje"
        "naive": Pipeline([("modelo", DummyRegressor(strategy="constant", constant=0.0))]),
        "ridge": Pipeline([("escala", StandardScaler()),
                           ("modelo", Ridge(alpha=1.0, random_state=SEMENTE))]),
        "gradient_boosting": Pipeline([("escala", StandardScaler()),
                                       ("modelo", GradientBoostingRegressor(
                                           n_estimators=300, learning_rate=0.03, max_depth=3,
                                           subsample=0.8, random_state=SEMENTE))]),
    }

    resultados = {}
    for nome, pipe in candidatos.items():
        pipe.fit(X_tr, y_tr)
        resultados[nome] = avaliar(preco_te, y_te.to_numpy(), pipe.predict(X_te))
        m = resultados[nome]
        log(f"{nome:<18} RMSE US$ {m['rmse_preco_usd']:>8,.0f} | MAPE {m['mape_preco_pct']:>5.2f}%"
            f" | direção {m['acuracia_direcao_pct']:>5.1f}%")

    disputa = {k: v for k, v in resultados.items() if k != "naive"}
    escolhido = min(disputa, key=lambda k: disputa[k]["rmse_preco_usd"])
    rmse_naive = resultados["naive"]["rmse_preco_usd"]
    rmse_esc = resultados[escolhido]["rmse_preco_usd"]
    bateu = rmse_esc < rmse_naive
    ganho = 100 * (rmse_naive - rmse_esc) / rmse_naive
    log(f"escolhido: {escolhido} (menor RMSE entre os que aprendem)")
    log(f"vs. baseline Naive: {ganho:+.1f}%  ->  {'BATEU' if bateu else 'NÃO BATEU'}"
        "   <<< registrar no devlog")

    # -------------------- calibração da faixa (volatilidade EWMA)
    sigma_te = X_te["volatilidade_ewma"].to_numpy()
    calibracao = {}
    for nivel in NIVEIS:
        z = float(norm.ppf(0.5 + nivel / 200))
        dentro = np.abs(y_te.to_numpy()) <= z * sigma_te
        calibracao[f"nivel_{nivel}"] = {
            "z": round(z, 4),
            "cobertura_empirica_pct": round(float(dentro.mean() * 100), 2),
            "largura_media_pct": round(
                float(np.mean(np.exp(z * sigma_te) - np.exp(-z * sigma_te)) * 100), 2),
        }
        log(f"faixa {nivel}%: cobertura empírica "
            f"{calibracao[f'nivel_{nivel}']['cobertura_empirica_pct']:.1f}% "
            f"(largura média {calibracao[f'nivel_{nivel}']['largura_media_pct']:.1f}%)")

    # ------------------ re-treino com a série inteira, para exportar
    # A escolha foi feita olhando o teste; agora quero o modelo com a
    # informação mais recente possível antes de salvar.
    final = candidatos[escolhido]
    final.fit(X, y)
    log("re-treinado com 100% dos dados para exportação")

    est = final.named_steps["modelo"]
    if hasattr(est, "feature_importances_"):
        imp = dict(sorted(zip(ft.FEATURES, map(float, est.feature_importances_)),
                          key=lambda t: -t[1]))
    elif hasattr(est, "coef_"):
        imp = dict(sorted(zip(ft.FEATURES, map(float, est.coef_)), key=lambda t: -abs(t[1])))
    else:
        imp = {}
    log("features mais influentes: " + ", ".join(list(imp)[:4]))

    # ------------------------------------------------------------ artefato
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")

    artefato = {
        "pipeline": final,                    # scaler + modelo, juntos
        "features": ft.FEATURES,              # contrato com a API
        "minimo_historico": ft.MINIMO_HISTORICO,
        "lambda_ewma": ft.LAMBDA_EWMA,
        "tipo_predicao": "retorno_log_d1",
        "nome_modelo": escolhido,
        "versao_artefato": "1.0.0",
        "treinado_em": agora,
        "periodo_treino": [str(datas.iloc[0].date()), str(datas.iloc[-1].date())],
        "ultimo_preco_conhecido": float(d["preco_btc_usd"].iloc[-1]),
        "ultimo_juros_conhecido": float(d["juros_pct"].iloc[-1]),
        "ultima_data_conhecida": str(d["data"].iloc[-1].date()),
        "metricas_teste": resultados[escolhido],
        "metricas_naive": resultados["naive"],
        "bateu_baseline_naive": bool(bateu),
        "calibracao_faixa": calibracao,
        "versao_sklearn": sklearn.__version__,
        "versao_python": platform.python_version(),
    }
    caminho = ARTEFATOS / "modelo.joblib"
    joblib.dump(artefato, caminho, compress=3)
    log(f"ARTEFATO salvo: {caminho} ({caminho.stat().st_size/1024:.0f} KB)")

    # relatório legível, para colar no devlog
    relatorio = {
        "gerado_em": agora,
        "fonte_dados": str(DADOS),
        "linhas_dataset": int(len(d)),
        "exemplos_utilizaveis": int(len(X)),
        "periodo": artefato["periodo_treino"],
        "features": ft.FEATURES,
        "split": {
            "tipo": "cronologico_sem_shuffle",
            "proporcao_teste": PROP_TESTE,
            "treino_dias": int(corte),
            "teste_dias": int(len(X) - corte),
            "teste_periodo": [str(datas.iloc[corte].date()), str(datas.iloc[-1].date())],
        },
        "modelo_escolhido": escolhido,
        "criterio_escolha": "menor rmse_preco_usd no teste, entre os modelos que aprendem",
        "bateu_baseline_naive": bool(bateu),
        "ganho_sobre_naive_pct": round(ganho, 2),
        "metricas_por_modelo": resultados,
        "calibracao_faixa": calibracao,
        "importancia_features": imp,
        "ultimo_preco_conhecido": artefato["ultimo_preco_conhecido"],
        "ultima_data_conhecida": artefato["ultima_data_conhecida"],
    }
    (ARTEFATOS / "metricas.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"relatório salvo: {ARTEFATOS/'metricas.json'}")

    # exemplo de requisição pronto, com os últimos dias reais
    n = ft.MINIMO_HISTORICO + 9
    (ARTEFATOS / "exemplo_requisicao.json").write_text(json.dumps({
        "precos": d["preco_btc_usd"].tail(n).round(2).tolist(),
        "juros": d["juros_pct"].tail(n).round(4).tolist(),
        "data_final": artefato["ultima_data_conhecida"],
    }, indent=2), encoding="utf-8")
    log(f"exemplo de requisição salvo ({n} dias)")
    log("TREINAMENTO CONCLUÍDO")


if __name__ == "__main__":
    main()
