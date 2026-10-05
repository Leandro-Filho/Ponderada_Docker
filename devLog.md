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



