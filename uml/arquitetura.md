# Esboço UML da arquitetura

> Entregável 1 do enunciado: *"faça um esboço UML que represente os componentes
> da solução e a troca de dados entre eles. Inclua, no mínimo, o ambiente de
> treinamento, o artefato gerado pelo modelo, o container de inferência/backend
> e a aplicação cliente. **Indique como o modelo treinado chega ao container de
> inferência.**"*

**A resposta da pergunta em negrito, em uma frase:**

> O artefato chega por um **volume nomeado do Docker**. O container de treino
> monta o volume `artefatos` em modo escrita e grava `modelo.joblib`. O container
> de inferência monta **o mesmo volume em somente leitura** (`:ro`) e carrega o
> arquivo quando sobe. Os dois nunca se falam diretamente e não precisam estar no
> ar ao mesmo tempo — o contrato entre eles é o arquivo.

São **dois containers**, como o enunciado pede. A aplicação cliente é uma página
servida pela própria API, na mesma origem — assim não existe um terceiro
container nem problema de CORS.

---

## 1. Componentes — a visão geral

```mermaid
flowchart LR
    subgraph FONTES["dados brutos (no disco)"]
      B[("btc_bruto.csv<br/>Coin Metrics<br/>1826 dias")]
      J[("juros_bruto.csv<br/>FRED EFFR<br/>1824 dias")]
    end

    P["preparar.py<br/>LIMPEZA<br/>junta, corrige vazamento,<br/>forward-fill, valida"]
    DS[("dataset.csv<br/>1692 dias x 6 colunas")]

    subgraph TREINO["container 1: treino (job)"]
      T["treinar.py<br/>split cronológico<br/>modelo vs baseline Naive"]
    end

    VOL[("volume nomeado: artefatos<br/>modelo.joblib<br/>metricas.json")]

    subgraph API["container 2: api (serviço)"]
      A["Flask + gunicorn :8000<br/>carrega o artefato na subida<br/>+ serve a página"]
    end

    NAV(["navegador / curl / Postman<br/>= a aplicação cliente"])
    F["comum/features.py<br/>fonte única das features"]

    B --> P
    J --> P
    P --> DS
    DS -->|"bind mount :ro"| T
    T  -->|"joblib.dump — ESCRITA"| VOL
    VOL -->|"joblib.load — SOMENTE LEITURA"| A
    NAV <-->|"HTTP :8000"| A
    F -.->|"importado"| T
    F -.->|"importado"| A

    style VOL fill:#f7a93b,stroke:#b8761f,color:#1a1205
    style P fill:#3b7f6b,stroke:#2a5d4e,color:#f0fff9
    style F fill:#2d3440,stroke:#4a556b,color:#e8eaed
```

Três coisas para explicar na banca:

| Caixa | Por que ela importa |
|---|---|
| **laranja** (volume) | é a resposta da pergunta do enunciado: é por aqui que o modelo viaja |
| **verde** (`preparar.py`) | é a etapa de limpeza. Ela existe porque juntar BTC (7 dias/semana) com juros (só dia útil) deixa 31,6% das linhas sem juros observado |
| **cinza** (`features.py`) | é importado pelo treino **e** pela API. Se a conta das features fosse escrita duas vezes, elas divergiriam e o modelo receberia em produção números diferentes dos que viu treinando — *training/serving skew*. A API confere a lista de features do artefato na subida e acusa se diferir |

---

## 2. Sequência — o fluxo de uma predição

```mermaid
sequenceDiagram
    autonumber
    actor U as Cliente<br/>(navegador / curl)
    participant A as api (gunicorn)
    participant F as features.py
    participant M as modelo.joblib<br/>(em memória)

    Note over A,M: na SUBIDA do container:<br/>joblib.load() uma única vez

    U->>A: POST /prever<br/>{precos:[...], juros:[...]}
    activate A

    A->>A: valida o corpo<br/>(listas do mesmo tamanho? dias suficientes?)

    alt corpo inválido
        A--)U: 400 {erro: "histórico insuficiente: ..."}
    else corpo válido
        A->>F: calcular features do último dia
        F--)A: vetor de features
        A->>M: pipeline.predict(X)
        M--)A: retorno previsto
        A->>A: preço = preço_hoje × e^retorno<br/>faixa = ± z × volatilidade
        A--)U: 200 {preco_previsto, faixa, servido_por}
    end
    deactivate A
```

Notação: seta de **ponta cheia** = mensagem síncrona (quem chama espera a
resposta); seta **tracejada** = retorno; `alt` = caminho alternativo.

O caminho de erro está desenhado de propósito. Um diagrama de sequência que só
mostra o caso feliz está incompleto, e isso é fácil de perguntar na banca.

---

## 3. Implantação — onde cada coisa roda

```mermaid
flowchart TB
    subgraph HOST["«device» máquina do desenvolvedor"]
      D[("«artifact» dados/dataset.csv<br/>no disco do host")]

      subgraph ENGINE["«execution environment» Docker Engine"]
        subgraph REDE["rede do Compose (bridge)"]
          TR["«container» treino<br/>python:3.12-slim<br/>job, sem porta"]
          AP["«container» api<br/>python:3.12-slim + gunicorn<br/>:8000 → publicada em 8000"]
        end
        V[("«volume» artefatos<br/>«artifact» modelo.joblib<br/>«artifact» metricas.json")]
      end
    end

    NAV(["«device» navegador / curl"])

    D  -->|"bind mount :ro"| TR
    TR -->|"escrita"| V
    V  -->|"leitura (:ro)"| AP
    NAV -->|"HTTP tcp:8000"| AP

    style V fill:#f7a93b,stroke:#b8761f,color:#1a1205
```

Anotações que valem ponto:

| Anotação | Por quê |
|---|---|
| o `dataset.csv` entra como **bind mount `:ro`** | o treino lê o dado, nunca o altera; o CSV é a entrada versionada no Git |
| o volume é **`:ro` na api** | a inferência não pode corromper o artefato que o treino produziu |
| o treino **não tem porta** | é um job, não um serviço: roda, grava e termina com exit 0 |
| só a `api` publica porta | é o único componente que alguém de fora precisa alcançar |

---

## 4. Atividades — o ciclo de vida do modelo

```mermaid
flowchart TD
    A([início]) --> L["python dados/preparar.py<br/>gera dataset.csv"]
    L --> B["docker compose build"]
    B --> C["docker compose run --rm treino"]
    C --> D{"exit 0?"}
    D -- não --> E["docker compose logs treino"] --> C
    D -- sim --> F[("modelo.joblib<br/>no volume")]
    F --> G["docker compose up -d api"]
    G --> H{"GET /health<br/>modelo_carregado = true?"}
    H -- não --> I["503: artefato não encontrado<br/>→ rodar o treino"] --> C
    H -- sim --> J["POST /prever"]
    J --> K([predição demonstrada])

    F -.->|"re-treinar sem derrubar a api"| M["GET /recarregar"] --> H
```

O caminho pontilhado é o ganho concreto do desacoplamento: re-treinar e
recarregar **sem reiniciar** o container de inferência.

---

## 5. Como o artefato chega na inferência — as alternativas

O enunciado pede para *indicar* como o modelo chega ao container de inferência.
Existem quatro caminhos, e a escolha precisa de justificativa:

| Caminho | Como funciona | Prós | Contras | Usei? |
|---|---|---|---|---|
| **Volume nomeado** | treino grava, inferência lê o mesmo volume | re-treina sem rebuildar imagem; containers desacoplados; funciona offline | o volume é estado local, não versionado | ✅ **sim** |
| `COPY` na imagem | o artefato entra na imagem no build | imagem autocontida, boa para produção/CI | qualquer re-treino exige build e deploy novos |  não |
| Bind mount de pasta do host | `./artefatos:/artefatos` | dá para abrir o arquivo e inspecionar | depende do caminho e das permissões do host | não |
| Registry de modelos / S3 | a inferência baixa na subida | versiona o modelo, serve vários ambientes | precisa de rede, credencial e serviço externo | não |

**Justificativa:** dentro do tempo da atividade, o volume nomeado dá o
desacoplamento que o enunciado quer demonstrar sem introduzir dependência de
rede, credencial ou rebuild. E é o caminho que torna o fluxo **visível** —
que é o próximo item.

---

## 6. A demonstração que prova o diagrama

Em vez de só apontar para o desenho, dá para mostrar o artefato viajando:

```bash
docker compose down -v          # apaga o volume → o artefato morre
docker compose up -d api
curl -s localhost:8000/health   # modelo_carregado: false
curl -s localhost:8000/prever   # 503 modelo indisponível
docker compose run --rm treino  # re-treina, grava no volume
curl -s localhost:8000/recarregar
curl -s localhost:8000/prever   # volta a 200
```

Se o backend responde `503` quando o volume está vazio e `200` depois do treino,
está provado na frente de quem avalia que o modelo **passa por ali** e não está
embutido na imagem.

---

## Como exportar para imagem

Os blocos acima são Mermaid, e o GitHub renderiza direto no README. Para PNG no
slide:

1. <https://mermaid.live> — colar o bloco e exportar PNG/SVG; ou
2. `npx -y @mermaid-js/mermaid-cli -i uml/arquitetura.md -o uml/diagrama.png`
