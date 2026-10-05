1. pegar o dataset

uma coisa que eu queria fazer é prever o valor do bitcoin com base em notícias e tudo mais. Então vou pegar um dataset dos valor dos bitcoins.

O que vou fazer aqui é pedir para o claude me gerar um dataset com base no valor do bitcoin, com valores diários de 5 anos atrás até agora, com base no Yahoo Finance. Com isso, teremos uma base muito boa para trabalhar.

2. começar a pensar no escopo da arquitetura do projeto

o que eu penso que seria ideal é o seguinte: sabemos que deverá ser um modelo que veja padrões, já que o bitcoin sobe quando os juros americanos estão baixos ou alguem aporta muito dinheiro nele, então precisamos de um modelo que acompanhe isso para nós. 

penso que deverá ser um Ramdom Forest ou um XGBoost. para tirar a conclusão, vou pegar com o claude qual ele acha que é mais adequado, além de começar a puxar dados dos juros americanos e outras features que afetam o valor do bitcoin para o projeto.

Estou indo no investing.com para pegar o dataset dos juros americanos. Deu problema no email e estou mudando ele. Não deu certo, vou pedir para alguma ia gerar para mim. No caso, o claude está gerando o dataset dos preços do bitcoin e o GPT está gerando o juros americanos.

Por algum motivo, o claude esqueceu de dar o dataset dos valores do bitcoin, mas o gpt já nos deu o juros americanos

A arquitetura em si, pensei no seguite: dois containers com api e treinamento do modelo, com essa estrutura: dados brutos → preparar.py (a limpeza) → treinar.py → modelo.joblib no volume → app.py lê e serve → testar.sh prova

Ou seja, vamos containizar o treino e a api, já esses são as pastas que rodam o projeto como um todo.

outro ponto que mudei de ideia: o gpt tinha nos dado um dataset dp Fed, que altera o juros americanos de pouco em poucos, então vamos usar outra dataset dos juros americanos negociados no mercado porque ele muda diariamente

3. preparar o ambiente

vou começar a criar um .venv para as bibliotecas que precisaremos nesse projeto, já que meu computador é linux, então não podemos rodar o projeto sem .venv

gerei a limpeza dos dados na pasta dados e preparar.py. nela, está sendo feita a retirada de duplicadas, normalizando as colunas, separando quais colunas podem ser possíveis vazamentos para o modelo, tirando nulos, etc. Ou seja, uma limpeza e preparação dos dados completa.

além disso, gerei  diagrama uml de como deverá ser feita a aplicação: primeiro fazemos um post para pegar a predição do bitcoin e ele bate na api e verifica se os dados são suficientes ou não para isso, depois ele manda para a fetures.py calcular os preços dos último dia e retorna para a api. depois disso, a api manda para o modelo.joblib para prever o valor e ele retorna o resultado para a api que manda de volta para o cliente o resultado 200 e suas predições. aqui, precisei usar o claude tanto para gerar o diagrama quanto para me ajudar a fazer a estrutura, já que para mim fazia sentido colocar no container a preparação, mas com algumas conversas com ele, decidi manter com dois containers: um com o treino e outro com a api

a conversa entre os containers funcionará da seguinte forma: o dataset.csv está no meu computador e ele irá para o container do treino , com o treno feito, ele irá para o modelo.joblib e metricas.json e volta novamente para o docker engine para o container da api que manda o resultado previsto para a porta 8000, podendo ser acessada pelo curl por http

Próximo passo, será fazer o modelo.

4. fazer o código do modelo

com ajuda do claude, estamos fazendo o modelo em si, quen usou até o naive para ver se o modelo será adequado com o que temos.Como tenho pouco tempoo, vou deixar esse código e ver como ele se desempenha. Ele será rodado da seguinte forma:


```bash
pip install -r treino/requirements.txt
python dados/preparar.py                              # limpeza
DADOS=dados/dataset.csv ARTEFATOS=artefatos \
  python treino/treinar.py                            # treino → modelo.joblib
pytest testes/ -v                                     # 14 testes
```

o arquivo gerado usa a biblioteca scikit-learn para pegar os modelos e fazemos uma comparação de 3 modelos para ver qual se saí melhor e quando fui rodar para pegar as métricas, deu erro de chaves de api pelo artefato e voi investigar o porque. erro foi ocasionado porque a preparação foi feita com o python do .venv e o pytest foi feito com o python da máquina. Vou arrumar isso e vamos seguir. Deu certo e o treino foi feito. agora, vamos fazer a api e suas endpoints para fechar o projeto.

5. api do projeto 

a api foi feita com ajuda do claude. Nela segue bem o que fizemos no diagrama uml: como está no diagrama, temos 4 passos: valida o corpo, calcula as features, retorna a predição e monta a rewposta. comecei a fazer o dockerfile do api, com a seguinte lógica: python3.12, pega o comum/comum (que é as features do dataset) e o app.py, estando na porta 8000 e cdm foi uma recomendação do claude: o servidor WSGI de produção, sobe 2 processos worker, escuta em todas as interfaces, porta 8000, o - significa stdout e módulo app (arquivo app.py), variável app (a instância Flask). 

6. subir o projeto nos containers 

agora, vamos subir os projetos nos docker: 

rodamos:

```bash
docker compose up --build
```

o docker compose já sobe os dois containers que fizemos. Vou testar as imagens com docker ps para ver se elas estão rodando e está rodando 
```bash
b4e05f0beb43   ponderada_docker-api               "gunicorn -w 2 -b 0.…"   3 minutes ago   Up 3 minutes                  0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   ponderada_docker-api-1
```

agora, vamos testar as rotas como estão: 
primeiro, testaremos ops índices das rotas:

```bash
curl -s localhost:8000 | python3 -m json.tool
```

resultado satisfatório:

```bash 
(.venv) bebaran@bebaran-Nitro-ANV15-51:~/Área de trabalho/Ponderada_Docker$ curl -s localhost:8000 | python3 -m json.tool
{
    "minimo_historico": 61,
    "modelo_carregado": true,
    "rotas": {
        "GET /exemplo": "predi\u00e7\u00e3o com os \u00faltimos dias reais",
        "GET /health": "o servi\u00e7o est\u00e1 vivo",
        "GET /info": "qual modelo, m\u00e9tricas e calibra\u00e7\u00e3o",
        "GET /recarregar": "rel\u00ea o artefato sem reiniciar",
        "POST /prever": "{'precos': [...], 'juros': [...], 'data_final': 'AAAA-MM-DD'}"
    },
    "servico": "api-inferencia-btc"
}
```

agora, testaremos o modelo carregado:
```bash
curl -s localhost:8000/health | python3 -m json.tool
```

resposta ótima:
```bash
{
    "modelo_carregado": true,
    "servido_por": "b4e05f0beb43",
    "status": "ok"
}
```

quer dizer que o modelo está tudo bem e os containers está rodando

agora, vamos testar a predição: 
```bash
curl -s localhost:8000/exemplo | python3 -m json.tool
```

resultado bem legal:
```bash
{
    "entrada": {
        "data_final": "2026-05-23",
        "juros": [
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.64,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.63,
            3.62,
            3.62,
            3.62,
            3.62,
            3.62
        ],
        "precos": [
            72712.15,
            74723.96,
            74086.01,
            71253.96,
            69948.9,
            70510.42,
            69707.29,
            68018.15,
            70726.21,
            70610.02,
            71281.3,
            68722.97,
            66214.17,
            66351.07,
            65966.75,
            66651.23,
            68214.87,
            68116.82,
            66908.32,
            66891.74,
            67295.13,
            68921.81,
            68742.91,
            72057.12,
            71085.06,
            71788.68,
            72945.07,
            73119.11,
            70685.37,
            74632.75,
            74173.51,
            74761.96,
            75095.74,
            77114.48,
            75792.06,
            73905.2,
            75814.15,
            76107.44,
            78317.74,
            78233.68,
            77444.27,
            77624.71,
            78538.77,
            77256.3,
            76295.53,
            75795.01,
            76299.62,
            78132.64,
            78712.54,
            78649.17,
            79826.11,
            80936.75,
            81364.62,
            79989.0,
            80179.69,
            80663.63,
            82256.78,
            81714.74,
            80525.56,
            79291.91,
            81175.83,
            79063.41,
            78164.5,
            77497.7,
            76975.91,
            76807.46,
            77408.17,
            77595.39,
            75567.54,
            76619.87
        ]
    },
    "resultado": {
        "aviso": "Predi\u00e7\u00e3o experimental, para fins acad\u00eamicos. N\u00e3o \u00e9 recomenda\u00e7\u00e3o de investimento.",
        "data_prevista": "2026-05-24",
        "dias_de_historico_recebidos": 70,
        "direcao": "baixa",
        "faixas": {
            "80%": {
                "cobertura_medida_no_teste_pct": 82.57,
                "maximo": 78178.86,
                "minimo": 75076.4
            },
            "95%": {
                "cobertura_medida_no_teste_pct": 93.58,
                "maximo": 79021.27,
                "minimo": 74276.04
            }
        },
        "juros_atual_pct": 3.62,
        "modelo": "ridge",
        "preco_atual": 76619.87,
        "preco_previsto": 76611.92,
        "regime_juros": "parado",
        "retorno_log_previsto": -0.000104,
        "servido_por": "b4e05f0beb43",
        "variacao_prevista_pct": -0.0104,
        "volatilidade_diaria_estimada_pct": 1.58
    }
}
```

pronto!!! o projetp está nos containers bonitinhos e feitos. tem algumas coisas que eu não gostei que o modelo nem bate o naive, uma série temporal que simplesmente replica o dado passado, então para que o modelo seja mais robusto, deveria ter trabalhado mais nele.

útima coisa que o claude me aconselhou é fazer esse seguinte teste: 
```bash
docker compose run --rm treino 2>&1 | grep RMSE
docker compose run --rm treino 2>&1 | grep RMSE
```

isso vai ver se o treino é reprodutivel

o resultado foi esse:
```bash
docker compose run --rm treino 2>&1 | grep RMSE
[treino] naive              RMSE US$    1,969 | MAPE  1.61% | direção   0.0%
[treino] ridge              RMSE US$    1,990 | MAPE  1.62% | direção  49.5%
[treino] gradient_boosting  RMSE US$    2,165 | MAPE  1.75% | direção  48.3%
[treino] escolhido: ridge (menor RMSE entre os que aprendem)
[treino] naive              RMSE US$    1,969 | MAPE  1.61% | direção   0.0%
[treino] ridge              RMSE US$    1,990 | MAPE  1.62% | direção  49.5%
[treino] gradient_boosting  RMSE US$    2,165 | MAPE  1.75% | direção  48.3%
[treino] escolhido: ridge (menor RMSE entre os que aprendem)
```

