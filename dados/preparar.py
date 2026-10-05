"""
LIMPEZA — a caixa verde do diagrama de componentes.
 
    btc_bruto.csv  ─┐
                    ├─→ preparar.py ─→ dataset.csv
    juros_bruto.csv ┘
 
Cada passo do diagrama é uma função aqui embaixo, na mesma ordem:
 
    carregar_btc()          lê o preço do BTC
    carregar_juros()        lê a taxa e descarta as colunas inúteis
    detectar_vazamento()    acha dias fechados com a taxa do dia útil SEGUINTE
    corrigir_forward_fill() refaz o preenchimento olhando só para trás
    juntar()                interseção das duas séries
    classificar_regime()    subindo / parado / cortando
    validar()               trava se algo estiver errado
 
O PROBLEMA desta etapa: BTC negocia 7 dias por semana; juros só em dia útil.
Ao juntar, ~32% das linhas ficam sem taxa observada.
 
A REGRA: forward-fill é correto (na manhã de sábado só se sabe a taxa de
sexta). Backward-fill é VAZAMENTO — traria a taxa de segunda para o sábado
anterior, ou seja, informação do futuro.
 
Uso:  python dados/preparar.py
"""
from __future__ import annotations
 
import json
from pathlib import Path
 
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
 
AQUI = Path(__file__).resolve().parent
JANELA_REGIME = 60      # dias; mesma janela usada em comum/features.py
LIMIAR_REGIME = 0.10    # pontos percentuais
 
 
def log(m: str) -> None:
    print(f"[limpeza] {m}", flush=True)
 
 
def mercado_fechado(datas: pd.Series) -> pd.Series:
    """True onde não existe taxa observada: fim de semana ou feriado federal."""
    feriados = USFederalHolidayCalendar().holidays(start=datas.min(), end=datas.max())
    return (datas.dt.dayofweek >= 5) | datas.isin(feriados)
 
 
# ---------------------------------------------------------------- 1. carregar
def carregar_btc(rel: dict) -> pd.DataFrame:
    btc = pd.read_csv(AQUI / "btc_bruto.csv", parse_dates=["data"]).sort_values("data")
    log(f"BTC: {len(btc)} linhas | {btc['data'].min().date()} a {btc['data'].max().date()}")
 
    rel["btc"] = {
        "linhas": len(btc),
        "duplicatas": int(btc["data"].duplicated().sum()),
        "nulos": int(btc["preco_btc_usd"].isna().sum()),
        "precos_nao_positivos": int((btc["preco_btc_usd"] <= 0).sum()),
        "dias_faltando": len(pd.date_range(btc["data"].min(), btc["data"].max())) - len(btc),
    }
    log("BTC: " + " ".join(f"{k}={v}" for k, v in rel["btc"].items() if k != "linhas"))
    return btc.drop_duplicates("data").dropna(subset=["preco_btc_usd"]).reset_index(drop=True)
 
 
def carregar_juros(rel: dict) -> pd.DataFrame:
    bruto = pd.read_csv(AQUI / "juros_bruto.csv")
    col_data = "date" if "date" in bruto.columns else bruto.columns[0]
    col_taxa = next(c for c in bruto.columns if "rate_pct" in c or c.upper() == "EFFR")
    log(f"juros: {len(bruto)} linhas, {len(bruto.columns)} colunas no arquivo bruto")
 
    # descarta o que é constante, redundante ou derivável da data
    descartadas = []
    for c in bruto.columns:
        if c in (col_data, col_taxa):
            continue
        if bruto[c].nunique() <= 1:
            descartadas.append(f"{c} (constante)")
        elif pd.api.types.is_numeric_dtype(bruto[c]) and np.allclose(
                bruto[c].to_numpy(float), bruto[col_taxa].to_numpy(float) / 100):
            descartadas.append(f"{c} (= {col_taxa}/100, redundante)")
        else:
            descartadas.append(f"{c} (derivável da data)")
    log(f"juros: descartando {len(descartadas)} colunas -> {descartadas}")
    rel["colunas_descartadas"] = descartadas
 
    j = bruto[[col_data, col_taxa]].copy()
    j.columns = ["data", "juros_pct"]
    j["data"] = pd.to_datetime(j["data"])
    j["juros_pct"] = pd.to_numeric(j["juros_pct"], errors="coerce")   # o '.' do FRED vira NaN
    return j.drop_duplicates("data").sort_values("data").reset_index(drop=True)
 
 
# ------------------------------------------------------- 2. detectar e corrigir
def detectar_vazamento(j: pd.DataFrame, fechado: pd.Series, rel: dict) -> None:
    """Para cada dia de mercado fechado, compara o valor do arquivo com o dia
    útil ANTERIOR e com o SEGUINTE. Se bate com o seguinte e não com o
    anterior, o valor veio do futuro."""
    datas = j["data"].to_numpy()
    valores = j["juros_pct"].to_numpy(float)
    abertos = np.flatnonzero(~fechado.to_numpy())
 
    vazamentos = []
    for i in np.flatnonzero(fechado.to_numpy()):
        antes, depois = abertos[abertos < i], abertos[abertos > i]
        if antes.size == 0 or depois.size == 0:
            continue
        v, a, p = valores[i], valores[antes[-1]], valores[depois[0]]
        if np.isfinite([v, a, p]).all() and v != a and v == p:
            vazamentos.append({"data": str(pd.Timestamp(datas[i]).date()),
                               "dia_util_anterior": round(float(a), 4),
                               "valor_no_arquivo": round(float(v), 4),
                               "dia_util_seguinte": round(float(p), 4)})
 
    log(f"juros: {len(vazamentos)} dia(s) de mercado fechado com valor do dia útil SEGUINTE")
    for v in vazamentos[:5]:
        log(f"         {v['data']}: anterior={v['dia_util_anterior']} "
            f"arquivo={v['valor_no_arquivo']} seguinte={v['dia_util_seguinte']}  <- vazamento")
    rel["vazamento_backfill"] = {"dias_afetados": len(vazamentos), "detalhe": vazamentos}
 
 
def corrigir_forward_fill(j: pd.DataFrame, fechado: pd.Series) -> pd.DataFrame:
    """Joga fora TODO valor de dia fechado e refaz repetindo o último dia útil."""
    j = j.copy()
    j["juros_observado"] = (~fechado).astype(int)
    j.loc[fechado, "juros_pct"] = np.nan
    j["juros_pct"] = j["juros_pct"].ffill()          # só o passado
    return j.dropna(subset=["juros_pct"]).reset_index(drop=True)
 
 
# ------------------------------------------------------------------- 3. juntar
def juntar(btc: pd.DataFrame, juros: pd.DataFrame, rel: dict) -> pd.DataFrame:
    d = btc.merge(juros, on="data", how="inner")
    log(f"interseção BTC x juros: {len(d)} dias | {d['data'].min().date()} a {d['data'].max().date()}")
    rel["overlap"] = {"dias": len(d), "inicio": str(d["data"].min().date()),
                      "fim": str(d["data"].max().date())}
    return d
 
 
def classificar_regime(d: pd.DataFrame) -> pd.DataFrame:
    """Para onde o Fed está indo, pela variação da taxa nos últimos 60 dias.
    Olha só para trás. Os primeiros 60 dias ficam 'indefinido' em vez de
    receberem um palpite."""
    d = d.copy()
    variacao = d["juros_pct"] - d["juros_pct"].shift(JANELA_REGIME)
    d["regime_juros"] = np.select(
        [variacao > LIMIAR_REGIME, variacao < -LIMIAR_REGIME],
        ["subindo", "cortando"], default="parado")
    d.loc[variacao.isna(), "regime_juros"] = "indefinido"
    return d
 
 
# ------------------------------------------------------------------ 4. validar
def validar(d: pd.DataFrame) -> None:
    """Trava o script se qualquer invariante quebrar. Melhor falhar aqui do
    que treinar em cima de dado torto."""
    assert d["data"].is_monotonic_increasing, "datas fora de ordem"
    assert not d["data"].duplicated().any(), "data duplicada"
    assert (d["preco_btc_usd"] > 0).all(), "preço não positivo"
    assert d[["preco_btc_usd", "juros_pct"]].notna().all().all(), "nulo nas colunas principais"
    vazio = len(pd.date_range(d["data"].min(), d["data"].max())) - len(d)
    assert vazio == 0, f"{vazio} dia(s) faltando na sequência"
    log("validações: OK (ordenado, sem duplicata, sem nulo, sem buraco, preços positivos)")
 
 
# ---------------------------------------------------------------------- juntar
def main() -> None:
    rel: dict = {}
 
    btc = carregar_btc(rel)
    juros = carregar_juros(rel)
 
    fechado = mercado_fechado(juros["data"])
    detectar_vazamento(juros, fechado, rel)
    juros = corrigir_forward_fill(juros, fechado)
 
    d = juntar(btc, juros, rel)
    d = classificar_regime(d)
    validar(d)
 
    d = d[["data", "preco_btc_usd", "volume_btc_usd",
           "juros_pct", "juros_observado", "regime_juros"]]
    d.to_csv(AQUI / "dataset.csv", index=False, float_format="%.6f")
    log(f"dataset.csv: {len(d)} linhas x {len(d.columns)} colunas")
 
    preenchido = (1 - d["juros_observado"].mean()) * 100
    log(f"{preenchido:.1f}% das linhas têm juros PREENCHIDO (mercado fechado) — "
        "a coluna juros_observado marca quais")
 
    # resumo em linguagem humana — é o que vai direto para o devlog
    problemas = []
    if any(v for k, v in rel["btc"].items() if k != "linhas"):
        problemas.append("BTC tem duplicata, nulo, preço não positivo ou buraco de data")
    if rel["vazamento_backfill"]["dias_afetados"]:
        problemas.append(
            f"{rel['vazamento_backfill']['dias_afetados']} dias de mercado fechado traziam o "
            "juros do dia útil SEGUINTE (informação do futuro) — corrigido por forward-fill")
    if rel["colunas_descartadas"]:
        problemas.append(f"{len(rel['colunas_descartadas'])} colunas inúteis no arquivo de juros")
 
    rel.update({
        "problemas_encontrados": problemas,
        "pct_juros_preenchido": round(float(preenchido), 2),
        "distribuicao_regime": d["regime_juros"].value_counts().to_dict(),
        "linhas_finais": len(d),
        "colunas_finais": list(d.columns),
        "decisoes": [
            "mantive só data e taxa do arquivo de juros; o resto era constante, "
            "redundante ou derivável da data",
            "zerei todo valor de dia fechado (fim de semana e feriado federal) e refiz "
            "por forward-fill; nunca backward-fill, que seria vazamento",
            f"regime = variação do juros nos últimos {JANELA_REGIME} dias, limiar de "
            f"{LIMIAR_REGIME} p.p.; os primeiros {JANELA_REGIME} dias ficam 'indefinido'",
        ],
    })
    (AQUI / "relatorio_limpeza.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    log("relatorio_limpeza.json salvo — use no devlog")
    log("LIMPEZA CONCLUÍDA")
 
 
if __name__ == "__main__":
    main()
 
