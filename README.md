Armazenar os algoritmos de Machine Learning aprendidos durante a Pós - Ciência de Dados (PUC MG) e os respectivos cursos online (Minerando dados / Data Science Academy / Udemy...). 

## Algoritmo Regressão Linear

Utilizei o algoritmo de Regressão Linear para predição de preços do fechamento de uma ação.

Os dados utilizados são da ação PETR4 (Petrobras) dos anos de 2010 até 2017. Nessa base de dados já temos
a classe (fechamento). 

O objetivo é separar os dados de teste e treino para prever os respectivos valores de fechamento nos dados de teste.

Após a execução do modelo, verificar se os valores preditos estão iguais ou próximos aos valores reais.

##### O que aprendi com esse algoritmo:

* Biblioteca Scikit Learn / Pandas / Matplotlib / Datetime
* Técnicas de validação do modelo com o RSME - Root Mean Square Error
* Alteração de tipo de coluna usando datetime
* Visualização de dados utilizando Matplotlib
* Separação de dados do conjunto de dados para teste e treino do modelo


Podemos concluir que o modelo de Regressão linear aproximou-se dos valores reais do conjunto de teste. Acredito que a combinação de outras técnicas de preparação de dados e aplicação de métricas pode melhorar o resultado obtido.

------------------------------------------------------------------------------------------------------------------------------------------


## Algoritmo Naive Bayes

Realizei uma análise de um conjunto de dados públicos extraídos do Tweet com informações relacionadas ao Governo de Minas Gerais. 

##### O Naive Bayes tem três modelos: 

* Bernoulli Naive Bayes: Trabalha com a matriz de presença de valores.
* Gaussian Naive Bayes:Calcula-se a média e o desvio padrão dos valores de entrada para cada classe.
* Multinomial Naive Bayes: Utiliza a frequência de termos.

##### O que aprendi com esse algoritmo:

* Biblioteca Scikit Learn / Pandas / Matplotlib
* Técnicas de Cross Validation / Matriz de confusão / Tunning / GridSearchCV
* Modelo Multinomial Naive Bayes
* Métricas de validação do modelo criado.

Após o aprendizado, observei que a alteração dos parâmetros pode influenciar em cada classe de forma isolada. Podemos obter uma melhoria em determinada classe. Vai depender do objetivo que desejamos alcançar.

O Algoritmo Navie Bayes é simples e podemos usá-lo como algoritmo BaseLine.

Eu realizei algumas análises de sentimentos utilizando a ferramenta KNIME. Existem várias etapas que poderiam ser utilizada no Naive.

------------------------------------------------------------------------------------------------------------------------------------------

## Algoritmo KNN - N Nearest Neighbors

Durante a aprendizagem do KNN utilizei duas bases de dados que são utilizadas para quem está começando
na área de dados.

A famosa base de dados Iris e Digitos.

Vale a pena uma leitura nesse artigo (https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)

##### O que aprendi com esse algoritmo:

* Biblioteca Scikit Learn / Pandas / Matplotlib / Seaborn / Numpy / Warning
* Técnicas de Cross Validation / Matriz de confusão / Tunning / GridSearchCV
* Métricas de validação do modelo criado.
* Visualização de dados com o Seaborn

O ponto alto dessa aprendizagem foi o tunning realizado no algoritmo. Desta forma, podemos obter o melhor
parâmetro K do respectivo modelo.

------------------------------------------------------------------------------------------------------------------------------------------

## Algoritmo K - Means

Algoritmo do tipo não supervisionado que tem como objetivo encontrar similaridades entre os dados e agrupá-los conforme o número de cluster passado pelo argumento k.

Nesse algoritmo, utilizei novamente a base de dados Iris. A base de dados é bem pequena e intuitiva. Muito boa para aprendizado de alguns algoritmos.

##### O que aprendi com esse algoritmo:

* Biblioteca Scikit Learn / Pandas / Matplotlib / Seaborn /
* Técnicas de Métricas de validação / Matriz de confusão / Agrupamento de dados
* Utilização do método Elbow para estimar o valor de K
* Visualização de dados com o Seaborn

Esse algoritmo só funciona com variávels numéricas. Desta forma, precisamos realizar algumas alterações necessária na base de dados antes de prosseguir com a aplicação do algoritmo.

------------------------------------------------------------------------------------------------------------------------------------------

## Algoritmo SVM 

O algoritmo SVM é aplicado na classificação de flores e na classificação de músicas do Spotify. O SVM será utilizado para criação de classificador que identifica possíveis músicas que um usuário poderia gostar. O objetivo é realizar o treino do algoritmo, validar o modelo e fazer tunning dos parâmetros do SVM.

Nesse algoritmo, utilizei novamente a base de dados Iris. A base de dados é bem pequena e intuitiva. Muito boa para aprendizado de alguns algoritmos.

##### O que aprendi com esse algoritmo:

* Biblioteca Scikit Learn / Pandas / Matplotlib / Seaborn / Numpy / Warnings
* Técnicas de Métricas de validação: Label Encoder / One Hot Encoding / Tunning / Pipeline / Get_dummies / Acurácia


Na aprendizagem desse algoritmo, podemos observar que a utilização do SVM retorna melhores parâmetros com a combinação de técnicas e tunnig nos dados. Realizei a transformação de variáveis categóricas em numéricas para execução do algoritmo.

Ele trabalha com variáveis numéricas.

------------------------------------------------------------------------------------------------------------------------------------------

