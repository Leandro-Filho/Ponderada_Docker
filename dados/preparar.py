"""
LIMPEZA E ALINHAMENTO DOS DADOS

Entra:  dados/btc_bruto.csv    (Coin Metrics — preço diário do BTC)
        dados/juros_bruto.csv  (FRED EFFR — taxa efetiva dos fed funds)
        dados/dgs10_bruto.csv  (OPCIONAL — Treasury 10 anos, se existir)

Sai:    dados/dataset.csv      (série única, alinhada, pronta para o modelo)
        dados/relatorio_limpeza.json  (o que foi feito, para o devlog)

O PROBLEMA CENTRAL desta etapa:
    BTC negocia 7 dias por semana. Juros só existe em dia útil, e nem em
    todo dia útil (feriado federal americano fecha o mercado). Ao juntar as
    duas séries, cerca de 30% das linhas não têm juros observado.

A REGRA que resolve, e a armadilha:
    forward-fill (repetir o último juros CONHECIDO) é correto — na manhã de
    sábado o mercado só sabe o juros de sexta.
    backward-fill é VAZAMENTO — traria o juros de segunda para o sábado
    anterior, ou seja, informação do futuro entrando no passado.

    O arquivo de juros que recebi JÁ VINHA com esse vazamento em 5 sábados.
    Este script detecta, reporta e corrige.

Uso:
    python preparar.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

AQUI = Path(__file__).resolve().parent
BTC = AQUI / "btc_bruto.csv"
JUROS = AQUI / "juros_bruto.csv"
DGS10 = AQUI / "dgs10_bruto.csv"          # opcional, entra se existir
SAIDA = AQUI / "dataset.csv"
RELATORIO = AQUI / "relatorio_limpeza.json"

# janela usada para classificar o regime de política monetária (dias corridos).
# Olha só para trás: é a variação dos últimos N dias, nunca dos próximos.
JANELA_REGIME = 60
LIMIAR_REGIME = 0.10        # em pontos percentuais


def log(m: str) -> None:
    print(f"[limpeza] {m}", flush=True)


def dias_sem_juros(datas: pd.Series) -> pd.Series:
    """True onde o mercado de juros está FECHADO: fim de semana ou feriado
    federal americano. Nesses dias não existe taxa observada."""
    feriados = USFederalHolidayCalendar().holidays(
        start=datas.min(), end=datas.max())
    fim_de_semana = datas.dt.dayofweek >= 5
    feriado = datas.isin(feriados)
    return fim_de_semana | feriado


def main() -> None:
    rel: dict = {"problemas_encontrados": [], "decisoes": []}

    # ---------------------------------------------------------------- BTC
    btc = pd.read_csv(BTC, parse_dates=["data"]).sort_values("data")
    log(f"BTC: {len(btc)} linhas | {btc['data'].min().date()} a {btc['data'].max().date()}")

    dup = int(btc["data"].duplicated().sum())
    nulos = int(btc["preco_btc_usd"].isna().sum())
    nao_pos = int((btc["preco_btc_usd"] <= 0).sum())
    buracos = len(pd.date_range(btc["data"].min(), btc["data"].max())) - len(btc)
    log(f"BTC: duplicatas={dup} nulos={nulos} precos<=0={nao_pos} dias faltando={buracos}")
    rel["btc"] = {"linhas": len(btc), "duplicatas": dup, "nulos": nulos,
                  "precos_nao_positivos": nao_pos, "dias_faltando": buracos}
    if dup or nulos or nao_pos or buracos:
        rel["problemas_encontrados"].append("BTC tem duplicata, nulo ou buraco de data")
    btc = btc.drop_duplicates("data").dropna(subset=["preco_btc_usd"])

    # -------------------------------------------------------------- JUROS
    jb = pd.read_csv(JUROS)
    col_data = "date" if "date" in jb.columns else jb.columns[0]
    col_taxa = next(c for c in jb.columns if "rate_pct" in c or c.upper() == "EFFR")
    log(f"juros: {len(jb)} linhas, {len(jb.columns)} colunas no arquivo bruto")

    # PROBLEMA 1 — colunas redundantes ou constantes
    descartadas = []
    for c in jb.columns:
        if c in (col_data, col_taxa):
            continue
        if jb[c].nunique() <= 1:
            descartadas.append(f"{c} (constante)")
        elif pd.api.types.is_numeric_dtype(jb[c]) and np.allclose(
                jb[c].to_numpy(dtype=float), jb[col_taxa].to_numpy(dtype=float) / 100):
            descartadas.append(f"{c} (= {col_taxa}/100, redundante)")
        else:
            descartadas.append(f"{c} (derivável da data)")
    log(f"juros: descartando {len(descartadas)} colunas -> {descartadas}")
    rel["colunas_descartadas"] = descartadas
    rel["decisoes"].append(
        "mantive só data e taxa; as outras eram constantes, redundantes ou deriváveis da data")

    j = jb[[col_data, col_taxa]].copy()
    j.columns = ["data", "juros_pct"]
    j["data"] = pd.to_datetime(j["data"])
    j["juros_pct"] = pd.to_numeric(j["juros_pct"], errors="coerce")  # '.' do FRED -> NaN
    j = j.drop_duplicates("data").sort_values("data").reset_index(drop=True)

    # PROBLEMA 2 — o vazamento: valor de fim de semana copiado do dia útil SEGUINTE
    fechado = dias_sem_juros(j["data"])

    # Comparo, para cada dia de mercado FECHADO, o valor que veio no arquivo
    # com o último dia útil ANTERIOR e com o próximo dia útil POSTERIOR.
    # Se bate com o posterior e não com o anterior, o valor veio do futuro.
    datas_arr = j["data"].to_numpy()
    valores = j["juros_pct"].to_numpy(dtype=float)
    idx_aberto = np.flatnonzero(~fechado.to_numpy())
    vazamentos = []
    for i in np.flatnonzero(fechado.to_numpy()):
        antes = idx_aberto[idx_aberto < i]
        depois = idx_aberto[idx_aberto > i]
        if antes.size == 0 or depois.size == 0:
            continue
        v, a, p = valores[i], valores[antes[-1]], valores[depois[0]]
        if np.isfinite([v, a, p]).all() and v != a and v == p:
            vazamentos.append({
                "data": str(pd.Timestamp(datas_arr[i]).date()),
                "dia_util_anterior": round(float(a), 4),
                "valor_no_arquivo": round(float(v), 4),
                "dia_util_seguinte": round(float(p), 4),
            })
    log(f"juros: {len(vazamentos)} dia(s) de mercado fechado com valor do dia útil SEGUINTE (backward-fill)")
    for v in vazamentos[:5]:
        log(f"         {v['data']}: anterior={v['dia_util_anterior']} arquivo={v['valor_no_arquivo']}"
            f" seguinte={v['dia_util_seguinte']}  <- vazamento")
    rel["vazamento_backfill"] = {"dias_afetados": len(vazamentos), "detalhe": vazamentos}
    if vazamentos:
        rel["problemas_encontrados"].append(
            f"{len(vazamentos)} dias de mercado fechado traziam o juros do dia útil SEGUINTE "
            "(informação do futuro) — corrigido por forward-fill")

    # CORREÇÃO — reconstruir: só dia útil é observado, o resto é forward-fill
    j["juros_observado"] = (~fechado).astype(int)
    j.loc[fechado, "juros_pct"] = np.nan          # joga fora o que foi preenchido errado
    j["juros_pct"] = j["juros_pct"].ffill()       # ffill: só o passado
    j = j.dropna(subset=["juros_pct"])            # descarta o começo, se abrir em dia fechado
    rel["decisoes"].append(
        "zerei todo valor de dia fechado (fim de semana e feriado federal) e refiz por forward-fill; "
        "nunca backward-fill, que seria vazamento")

    # --------------------------------------------------------- DGS10 (opcional)
    tem_dgs10 = DGS10.exists()
    if tem_dgs10:
        t = pd.read_csv(DGS10)
        t.columns = ["data", "dgs10_pct"] + list(t.columns[2:])
        t = t[["data", "dgs10_pct"]]
        t["data"] = pd.to_datetime(t["data"])
        t["dgs10_pct"] = pd.to_numeric(t["dgs10_pct"], errors="coerce")  # '.' = feriado
        t = t.drop_duplicates("data").sort_values("data")
        log(f"DGS10: {len(t)} linhas, {int(t['dgs10_pct'].isna().sum())} valores '.' (feriado) -> ffill")
        rel["dgs10"] = {"linhas": len(t), "valores_ponto": int(t["dgs10_pct"].isna().sum())}
    else:
        log("DGS10: arquivo não encontrado — seguindo só com o EFFR")
        log("       para incluir depois: salve o CSV do FRED como dados/dgs10_bruto.csv e rode de novo")
        rel["dgs10"] = None

    # ------------------------------------------------------------- JUNTAR
    d = btc.merge(j, on="data", how="inner")
    log(f"interseção BTC x juros: {len(d)} dias | {d['data'].min().date()} a {d['data'].max().date()}")
    rel["overlap"] = {"dias": len(d), "inicio": str(d["data"].min().date()),
                      "fim": str(d["data"].max().date())}

    if tem_dgs10:
        d = d.merge(t, on="data", how="left")
        d["dgs10_observado"] = d["dgs10_pct"].notna().astype(int)
        d["dgs10_pct"] = d["dgs10_pct"].ffill()
        d = d.dropna(subset=["dgs10_pct"])
        log(f"com DGS10: {len(d)} dias")

    # ------------------------------------------------ regime de política monetária
    # variação do juros nos últimos JANELA_REGIME dias. Olha só para trás.
    var = d["juros_pct"] - d["juros_pct"].shift(JANELA_REGIME)
    d["regime_juros"] = np.select(
        [var > LIMIAR_REGIME, var < -LIMIAR_REGIME],
        ["subindo", "cortando"], default="parado")
    d.loc[var.isna(), "regime_juros"] = "indefinido"   # primeiros 60 dias
    rel["decisoes"].append(
        f"regime = variação do juros nos últimos {JANELA_REGIME} dias, limiar de "
        f"{LIMIAR_REGIME} p.p.; os primeiros {JANELA_REGIME} dias ficam 'indefinido' em vez de "
        "receberem um palpite")

    # -------------------------------------------------------------- validar
    assert d["data"].is_monotonic_increasing, "datas fora de ordem"
    assert not d["data"].duplicated().any(), "data duplicada"
    assert (d["preco_btc_usd"] > 0).all(), "preço não positivo"
    assert d[["preco_btc_usd", "juros_pct"]].notna().all().all(), "nulo nas colunas principais"
    vazio = len(pd.date_range(d["data"].min(), d["data"].max())) - len(d)
    assert vazio == 0, f"{vazio} dia(s) faltando na sequência"
    log("validações: OK (ordenado, sem duplicata, sem nulo, sem buraco, preços positivos)")

    # ---------------------------------------------------------------- salvar
    cols = ["data", "preco_btc_usd", "volume_btc_usd", "juros_pct", "juros_observado"]
    if tem_dgs10:
        cols += ["dgs10_pct", "dgs10_observado"]
    cols += ["regime_juros"]
    d = d[cols]
    d.to_csv(SAIDA, index=False, float_format="%.6f")
    log(f"dataset.csv: {len(d)} linhas x {len(d.columns)} colunas")

    pct_ffill = (1 - d["juros_observado"].mean()) * 100
    log(f"{pct_ffill:.1f}% das linhas têm juros PREENCHIDO (mercado fechado) — a coluna "
        "juros_observado marca quais")
    rel["pct_juros_preenchido"] = round(float(pct_ffill), 2)
    rel["distribuicao_regime"] = d["regime_juros"].value_counts().to_dict()
    rel["linhas_finais"] = len(d)
    rel["colunas_finais"] = list(d.columns)

    RELATORIO.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"relatorio_limpeza.json salvo — use no devlog")
    log("LIMPEZA CONCLUÍDA")


if __name__ == "__main__":
    main()
