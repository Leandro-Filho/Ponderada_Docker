# Previsão do preço do Bitcoin — treino e inferência em containers

Atividade ponderada M7 — Engenharia de Computação, Inteli.
Enunciado: [Murilo-ZC/Atividade-Ponderada-M7-2026-EC](https://github.com/Murilo-ZC/Atividade-Ponderada-M7-2026-EC).

Estima o preço de fechamento do Bitcoin do dia seguinte a partir do histórico
do próprio BTC e do **juros americano**. O treino roda num container, exporta
o modelo como artefato, e um segundo container carrega esse artefato e serve
as predições.

---

## A pergunta que vai aparecer na banca

> *"Como o modelo treinado chega no container de inferência?"*

**Por um volume nomeado do Docker.** O container de treino monta o volume
`artefatos` em escrita e grava `modelo.joblib`. O container da API monta **o
mesmo volume em somente leitura** (`:ro`) e carrega o arquivo na subida. Os
dois nunca se falam diretamente e não precisam estar no ar ao mesmo tempo — o
contrato entre eles é o arquivo.

As quatro alternativas consideradas e a justificativa da escolha estão em
[uml/arquitetura.md](uml/arquitetura.md#5-como-o-artefato-chega-na-inferência--as-alternativas).

---

## Rodar

```bash
python dados/preparar.py              # 1. limpeza  -> dados/dataset.csv
docker compose build                  # 2. constrói as duas imagens
docker compose run --rm treino        # 3. treina   -> modelo.joblib no volume
docker compose up -d api              # 4. sobe a inferência
```

Abrir **<http://localhost:8000>**.

```bash
curl -s http://localhost:8000/health | python3 -m json.tool   # modelo_carregado: true
curl -s http://localhost:8000/exemplo | python3 -m json.tool  # a predição
```

Derrubar: `docker compose down` (mantém o artefato) ou `docker compose down -v`
(apaga o volume e obriga a re-treinar).

### Sem Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python dados/preparar.py
DADOS=dados/dataset.csv ARTEFATOS=artefatos python treino/treinar.py
python -m pytest testes/ -v                 # com 'python -m', não 'pytest' solto
```

---

## Os componentes

| # | Serviço | Base | Tipo | Porta |
|---|---|---|---|---|
| 1 | `treino` | `python:3.12-slim` | job (roda e morre) | nenhuma |
| 2 | `api` | `python:3.12-slim` + gunicorn | serviço | 8000, publicada |

A aplicação cliente é uma página servida pela **própria API**, na mesma origem.
Isso elimina um terceiro container e qualquer problema de CORS.

### A API

| Método | Rota | O que faz |
|---|---|---|
| GET | `/` | a página de demonstração |
| GET | `/health` | liveness; usado pelo `HEALTHCHECK`. Responde 200 mesmo sem modelo, e `modelo_carregado` diz a verdade |
| GET | `/info` | qual modelo, quando treinado, de qual período |
| GET | `/modelo` | métricas de todos os modelos, calibração, importância das features |
| GET | `/exemplo` | predição com os últimos dias reais — demo de um clique |
| POST | `/prever` | `{"precos": [...], "juros": [...], "data_final": "AAAA-MM-DD"}` |
| GET | `/recarregar` | relê o artefato **sem reiniciar** o container |

Mínimo de **61 dias** de histórico nas duas listas (o regime de juros olha os
últimos 60). Entrada inválida devolve `400` com a mensagem dizendo o que falta —
nunca `500`.

---

## O pipeline

```
btc_bruto.csv  ─┐
                ├─→ preparar.py ─→ dataset.csv ─→ treinar.py ─→ modelo.joblib ─→ api
juros_bruto.csv ┘    (limpeza)      1692 dias      (treino)      (no volume)
                                         ↑              ↑                        ↑
                                    comum/features.py ──┴────────────────────────┘
```

`comum/features.py` é importado **pelo treino e pela API**. Se a conta das
features fosse escrita duas vezes, elas divergiriam e o modelo receberia em
produção números diferentes dos que viu treinando — *training/serving skew*. A
API ainda compara a lista de features do artefato com a sua própria na subida e
recusa servir se diferirem.

### A limpeza achou um vazamento

O arquivo de juros vinha com **8 dias de mercado fechado carregando a taxa do
dia útil seguinte** — informação do futuro. `preparar.py` detecta, reporta e
corrige por forward-fill. Detalhes em [dados/DICIONARIO.md](dados/DICIONARIO.md).

---

## O resultado honesto

| Modelo | RMSE | MAPE | Acerto de direção |
|---|---|---|---|
| **Naive** (amanhã = hoje) | **US$ 1.969** | **1,61%** | — |
| Ridge *(o escolhido)* | US$ 1.990 | 1,62% | 49,5% |
| Gradient Boosting | US$ 2.165 | 1,75% | 48,3% |

**Nenhum modelo bateu o baseline Naive.** Também testei SES, Holt e ARIMA na
mesma janela e nenhum superou o passeio aleatório — ARIMA(0,1,0), aliás, **é**
o passeio aleatório.

Três diagnósticos sustentam isso:

- **R² no teste: −0,012** (negativo = pior que prever a média);
- as previsões têm desvio-padrão de 0,167% contra 2,240% do retorno real — o
  modelo quase não se move;
- **teste placebo:** treinando com o alvo **embaralhado**, o RMSE é US$ 1.981,
  praticamente igual ao do modelo real (US$ 1.990). Isso prova as duas coisas de
  uma vez: **não há vazamento** (senão o real seria muito melhor) e **não há
  sinal** a extrair.

### O que funciona: a faixa de incerteza

| Faixa nominal | Cobertura medida no teste |
|---|---|
| 80% | **82,6%** |
| 95% | **93,6%** |

Vem da volatilidade EWMA (λ = 0,94, padrão RiskMetrics). A direção é
imprevisível; a **intensidade** não é. Por isso a API devolve a faixa junto com
o ponto.

---

## Testar

```bash
python -m pytest testes/ -v        # 14 testes
```

O mais importante é `test_features_nao_olham_o_futuro`: calcula as features com
a série inteira e depois com a série **cortada** no dia t, e exige que a linha
do dia t seja idêntica. Se alguma feature espiasse o futuro, divergiriam.

### A demonstração que prova o diagrama

```bash
docker compose down -v                      # apaga o volume: o artefato morre
docker compose up -d api
curl -s localhost:8000/health               # modelo_carregado: false
curl -s localhost:8000/exemplo              # 503 modelo indisponível
docker compose run --rm treino              # re-treina
curl -s localhost:8000/recarregar
curl -s localhost:8000/exemplo              # volta a 200
```

Se a API responde `503` com o volume vazio e `200` depois do treino, está
provado na frente de quem avalia que o modelo **passa por ali** e não está
embutido na imagem.

---

## Estrutura

```
.
├── README.md
├── DEVLOG.md                 ⭐ escrito por mim (40% da nota)
├── docker-compose.yml        2 serviços + 1 volume
├── requirements-dev.txt      para rodar fora do Docker
├── .env.example
│
├── dados/
│   ├── btc_bruto.csv         Coin Metrics, 1826 dias
│   ├── juros_bruto.csv       FRED EFFR, intocado de propósito
│   ├── preparar.py           ⭐ a limpeza
│   ├── dataset.csv           1692 dias × 6 colunas
│   ├── relatorio_limpeza.json
│   └── DICIONARIO.md
│
├── comum/features.py         ⭐ fonte única das features
├── treino/{Dockerfile,requirements.txt,treinar.py}
├── api/{Dockerfile,requirements.txt,app.py,pagina.html}
├── testes/test_modelo.py     14 testes
└── uml/arquitetura.md        4 diagramas + PNG e SVG
```

---

## Decisões que valem explicar

**O alvo é o retorno, não o preço.** Prevendo o preço direto, basta copiar o
valor de ontem e o R² passa de 0,99 — parece excelente e não aprendeu nada.

**O split é cronológico.** `shuffle=True` em série temporal é vazamento.

**gunicorn, não `app.run()`.** O servidor de desenvolvimento do Flask não é para
produção. `-b 0.0.0.0:8000` é obrigatório: em `127.0.0.1` ninguém de fora do
container alcança.

**As versões estão travadas com `==` nos dois requirements.** O `joblib` amarra
o modelo à versão do scikit-learn que o criou; se treino e API divergirem, o
`joblib.load` pode avisar ou falhar.

**A API sobe mesmo sem artefato.** Se ela morresse, o container entraria em loop
de reinício e a mensagem de erro se perderia. Subindo e respondendo `503` com o
motivo, o problema fica legível.

---

> As predições são **experimentais e acadêmicas**. Não são recomendação de
> investimento.
