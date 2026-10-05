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



