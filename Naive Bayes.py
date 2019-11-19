
# coding: utf-8

# In[ ]:


#Importação das bibliotecas que serão utilizadas na aprendizagem do Naive Bayes.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict


# In[ ]:


#Importando o arquivo contendo os dados que serão analisados durante a aprendizagem.

dados = pd.read_csv('tweets_mg.csv',encoding='utf-8')


# In[ ]:


# Usando o método info() para retornar informações sobre o conjunto de dados.
# Podemos observar no retorno do método todas as colunas e informações do conjunto de dados.

dados.info()


# In[ ]:


# Utilizamos o pacote pandas para visualizar o tamanho total da coluna "Text" que contém os tweets.
# Estou exibindo apenas os 100 primeiros registros através do parâmetro passado para o head().
# Mais informações sobre o set_option e as opções de parâmetros no link abaixo.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html 

pd.set_option('display.max_colwidth', -1)
dados.Text.head(100)


# In[ ]:


# Existem linhas duplicadas no conjunto de dados.
# Com o método drop_duplicates() podemos remover os dados de uma determinada colunaa do conjunto de dados.
# Devemos passar a coluna desejada dentro do colchetes e entre aspas simples.

dados.drop_duplicates(['Text'],inplace=True)


# In[ ]:


# Vamos contar a quantidade de dados existentes no conjunto após a exclusão de registros duplicados.

dados.count()


# In[ ]:


# Após a remoção de linhas duplicadas, vamos remover colunas desnecessárias para nossa análise.
# A exclusão de informações é realizada com base na análise de cada pessoa. 
# A preparação da base de dados é uma parte fundamental no processo de análise de dados.
# Temos um loop For iterando no conjunto de dados e removendo a coluna "Unnamed".

for i in dados.columns.values:
    if i.startswith('Unnamed'):
        dados.drop(i, axis=1, inplace=True)
        print ('Colunas Deletadas:', i)


# In[ ]:


# Visualizando as colunas do conjuntos de dados após a exclusão das informações desnecessárias.
dados.columns


# In[ ]:


# Verificando apenas as informações da coluna "Classificação".
# A coluna Classificação possuí a classifiação feita para cada tweet de forma manual.
# A classificação está definida como Neutro, Positivo, Negativo.

dados.Classificacao


# In[ ]:


# Passando o matplotlib inline para visualizar o gráfico no próprio jupyter notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

# Plotando a coluna classificação no gráfico e contando a quantidade de cada classificação no conjunto de dados.
# Vamos plotar um gráfico do tipo bar. Podemos observar o parâmetro kind definido abaixo.

dados.Classificacao.value_counts().plot(kind='bar')


# In[ ]:


# Realizando uma separação dos tweets e suas respectivas classes.
# Estamos realizando a separação dos dados para realizar o treino posteriormente.
# Estamos armazenando uma coluna do conjunto de dados em cada variável criada.

tweets = dados['Text']
classes = dados['Classificacao']


# In[ ]:


# Gerando o modelo através do metódo CountVectorizer, passando o parâmetro "word".
# Estamos dizendo para o modelo analisar cada palavra com base no bag of words.
# Na variável freq_tweets armazenamos a transformação da variável tweets criada anteriormente.

vetor = CountVectorizer(analyzer="word")
freq_tweets = vetor.fit_transform(tweets)


# In[ ]:


# Podemos visualuzar o tipo da variável freq_tweets. O resulta é uma matriz esparsa.

type(freq_tweets)


# In[ ]:


# Exibindo o resulta da matriz esparsa criada anteriormente. 
# O conjunto de dados pos

freq_tweets.shape


# In[ ]:


# Vamos utilizar o modelo Multinomial do Naive Bayes.
# Existem os seguintes modelos no Naive Bayes (Bernoulli Naive Bayes, Gaussian Naive Bayes, Multinomial Naive Bayes)

modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)


# In[ ]:


# Vamos testar o modelo criado passando algumas instâncias simples. Foram criadas de forma aleatória.
# Uma variável testes para receber instâncias que serão testadas pelo nosso modelo posteriormente.

testes = ['Esse governo está no início e é uma droga',
          'Estou feliz com o governo de Minas esse ano. Acho que podemos esperar coisas boas',
          'O estado de Minas Gerais decretou calamidade financeira!!! Socorro !!!!',
          'A segurança desse país é uma droga',
          'O governador de Minas é mais uma vez do PT. Eh Brasil !!!!']


# In[ ]:


# Transforma os dados de teste em vetores de palavras.

freq_testes = vetor.transform(testes)


# In[ ]:


# Realizando a classificação com o modelo treinado anteriormente.
# Utilizando o loopt for para exibir os resultados de cada "frase" criada anteriormente.

for t, c in zip (testes,modelo.predict(freq_testes)):
    print (t +", "+ c)


# In[ ]:


# Probabilidades de cada classe do conjunto de dados.
# Usamos o predict_proba para visualizar os dados de probabilidade. 
# A visualização será um array com cada frase em sua respectiva posição.

print (modelo.classes_)
modelo.predict_proba(freq_testes).round(2)


# In[ ]:


# Vamos realizar uma avaliação do modelo utilizando a técnica de "Cross Validation"
# Utilizaremos com 10 folds. Mais informações no link abaixo:
# https://scikit-learn.org/stable/modules/cross_validation.html

resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)


# In[ ]:


# Utilizando a matriz de confusão. Essa técnica é excelente para validação.
# Temos como analisar os dados que foram interpretados em classes "incorretas".

print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True))


# In[ ]:


# Vamos umsar o método metrics do sklearn. Vale a pena uma leitura na documentação. 
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

sentimento=['Positivo','Negativo','Neutro']
print (metrics.classification_report(classes,resultados,sentimento))


# In[ ]:


# Vamos aplicar o tunning no algoritmo Naive Bayes.
# Usamos o tunning para executar o modelo por várias vezes alterando o parâmetro.
# Importar o GridSearchCV do pacote abaixo.

from sklearn.model_selection import GridSearchCV


# In[ ]:


# Criação de uma variável para receber um lista com range de 1 até 10.
# Iremos utilizar posteriormente a variável criada.

lista_alpha = list(range(1,11))
lista_alpha


# In[ ]:


# Criando um dicionário com o nome do parâmetro e a lista de valores.
# Repare que a chave do diicionário é o nome do parâmetro do modelo MultinomialNB().

parametros_grid = dict(alpha=lista_alpha)


# In[ ]:


# Instanciando o modelo através da variável nvModelo. Já criei uma variável modelo anteriormente.

nvModelo = MultinomialNB()


# In[ ]:


# Crianndo o objeto grid passando o metódo GridSearchCV com os respectivos parâmetros.
# Podemos observar que passei o modelo criado, o dicionário com o range de valores, quantidade de folds e o scoring.

grid = GridSearchCV(nvModelo, parametros_grid, cv=10, scoring='accuracy')


# In[ ]:


# Vamos executar o objeto grid criado anteriormente. 
# Ao executar o script abaixo, vamos ter como resultado os parâmetros. 
# Podemos visualizar o param_grid contendo os dados do dicionário criado anteriormente.

freq_tweets = vetor.fit_transform(tweets)
grid.fit(freq_tweets, classes)


# In[ ]:


# Vamos visualizar quais são os melhores scores com o metódo best_score_
# Documentação do GridSearchCV no link abaixo:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

grid.best_score_ 


# In[ ]:


# Verificando qual foi o melhor parâmetro. Com base no dicionário de parâmetros criados anteriormente.

grid.best_params_

