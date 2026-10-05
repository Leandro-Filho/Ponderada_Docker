"""
FONTE ÚNICA DE VERDADE DAS FEATURES.

Importado PELO TREINO e PELA API. É de propósito: se a conta das features
fosse escrita duas vezes, treino e inferência divergiriam com o tempo e o
modelo receberia em produção números diferentes dos que viu treinando.
Esse bug chama-se training/serving skew e é silencioso — o modelo não quebra,
só erra mais, e ninguém descobre por quê.

REGRA DE OURO: toda feature aqui olha SÓ PARA TRÁS.
Nenhuma usa o preço, o juros ou qualquer informação de amanhã.

São 10 features, agrupadas em três blocos:
  - 5 do preço do BTC
  - 4 do juros americano
  - 1 de calendário
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A ORDEM importa: é o contrato entre o treino e a API.
# A API confere esta lista contra a que veio no artefato e recusa se diferir.
FEATURES: list[str] = [
    # --- preço do BTC ---
    "retorno_log_1d",      # variação de ontem para hoje
    "retorno_log_7d",      # variação da última semana
    "dist_media_30d",      # quanto o preço está acima/abaixo da média de 30 dias
    "volatilidade_ewma",   # volatilidade recente, com peso maior nos dias próximos
    "rsi_14",              # índice de força relativa: 0 = só caiu, 100 = só subiu
    # --- juros americano ---
    "juros_pct",           # nível da taxa
    "var_juros_30d",       # quanto a taxa mudou em 30 dias
    "regime_subindo",      # 1 se o Fed está em ciclo de alta
    "regime_cortando",     # 1 se está em ciclo de corte
    # --- calendário ---
    "fim_de_semana",       # 1 em sábado e domingo (o mercado de juros está fechado)
]

# Mínimo de dias de histórico para TODAS as features nascerem completas.
# Quem manda é o regime, que olha a variação do juros nos últimos 60 dias.
MINIMO_HISTORICO: int = 61

LAMBDA_EWMA: float = 0.94     # padrão RiskMetrics para dados diários
JANELA_REGIME: int = 60       # mesma janela usada em dados/preparar.py
LIMIAR_REGIME: float = 0.10   # em pontos percentuais


def calcular(precos, juros, datas=None) -> pd.DataFrame:
    """Recebe as séries diárias (do mais ANTIGO para o mais RECENTE) e devolve
    o DataFrame de features, uma linha por dia.

    precos : fechamento do BTC em USD
    juros  : taxa de juros em % ao ano, já alinhada dia a dia com os preços
    datas  : opcional; se ausente, assume dias corridos terminando hoje
    """
    p = pd.Series(np.asarray(precos, dtype=float)).reset_index(drop=True)
    j = pd.Series(np.asarray(juros, dtype=float)).reset_index(drop=True)

    if len(p) != len(j):
        raise ValueError(f"precos e juros têm tamanhos diferentes: {len(p)} vs {len(j)}")

    if datas is None:
        datas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(p))
    datas = pd.to_datetime(pd.Series(np.asarray(datas))).reset_index(drop=True)

    r = np.log(p / p.shift(1))          # retorno logarítmico diário
    f = pd.DataFrame(index=p.index)

    # ---------------------------------------------------------- preço do BTC
    f["retorno_log_1d"] = r
    f["retorno_log_7d"] = np.log(p / p.shift(7))
    f["dist_media_30d"] = p / p.rolling(30).mean() - 1

    # EWMA dá peso maior aos dias recentes: reage rápido a mudança de regime
    # de volatilidade. É também o que alimenta a faixa de incerteza.
    f["volatilidade_ewma"] = np.sqrt(r.ewm(alpha=1 - LAMBDA_EWMA).var())

    delta = p.diff()
    ganho = delta.clip(lower=0).rolling(14).mean()
    perda = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + ganho / perda.replace(0, np.nan))

    # ----------------------------------------------------------------- juros
    f["juros_pct"] = j
    f["var_juros_30d"] = j - j.shift(30)

    # Regime: para onde o Fed está indo, olhando a variação dos últimos 60 dias.
    # Mesma regra do preparar.py — se mudar lá, tem de mudar aqui.
    var_regime = j - j.shift(JANELA_REGIME)
    f["regime_subindo"] = (var_regime > LIMIAR_REGIME).astype(float)
    f["regime_cortando"] = (var_regime < -LIMIAR_REGIME).astype(float)
    # os dois zerados = regime "parado" (categoria de referência)
    f.loc[var_regime.isna(), ["regime_subindo", "regime_cortando"]] = np.nan

    # ------------------------------------------------------------ calendário
    f["fim_de_semana"] = (datas.dt.dayofweek >= 5).astype(float)

    return f[FEATURES]


def ultima_linha(precos, juros, datas=None) -> pd.DataFrame:
    """Features do ÚLTIMO dia da série — é o que a inferência precisa.
    Valida a entrada e levanta ValueError com mensagem clara, para a API
    devolver 400 em vez de 500."""
    p = np.asarray(precos, dtype=float)
    j = np.asarray(juros, dtype=float)

    if len(p) != len(j):
        raise ValueError(f"precos e juros têm tamanhos diferentes: {len(p)} vs {len(j)}")
    if len(p) < MINIMO_HISTORICO:
        raise ValueError(
            f"histórico insuficiente: recebi {len(p)} dias, preciso de pelo menos "
            f"{MINIMO_HISTORICO} (o regime de juros olha os últimos {JANELA_REGIME} dias)")
    if not np.isfinite(p).all() or not np.isfinite(j).all():
        raise ValueError("há valor nulo ou não numérico nas séries")
    if (p <= 0).any():
        raise ValueError("preço precisa ser positivo (uso logaritmo nos retornos)")

    f = calcular(p, j, datas)
    linha = f.iloc[[-1]].copy()
    if linha.isna().any().any():
        faltando = linha.columns[linha.isna().iloc[0]].tolist()
        raise ValueError(f"features não calculáveis com esse histórico: {faltando}")
    return linha
