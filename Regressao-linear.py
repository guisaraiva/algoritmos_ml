
# coding: utf-8

# In[ ]:


# Importação das bibliotecas que serão utilizadas no estudo do algoritmo.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import plotly
plotly.offline.init_notebook_mode()
import datetime


# In[ ]:


# O conjunto de dados possui registrados dos preços da Petribrás entre 01/2010 até 11/2017.
# Vamos utilizar esses dados para criar nosso modelo preditivo e comparar os respectivos valores.
# Não irei utilizar o encoding, pois não temos dados textuais no conjunto de dados.

cjdados = pd.read_csv('petr4.csv')


# In[ ]:


# Transformando a coluna Date em uma coluna do tipo Datetime

cjdados['Date'] = pd.to_datetime(cjdados['Date'])


# In[ ]:


# Visualizando os dados com o método tail. O método tail() exibe os últimos registros.
# O método head() exibe os primeiros registros.

cjdados.tail()
#cjdados.head()


# In[ ]:


# Variação entre o preco de abertura e fechamento.
# Criando um coluna "Variação" para recebeer o resultado do valor de fechamento e valor de abertura.


cjdados['Variation'] = cjdados['Close'].sub(cjdados['Open'])


# In[ ]:


# Criar gráfico com os valores dos preços no periodo analisado 2010 a 2017.
# Utiliza a biblioteca pyplot para plotar dados financeiros temporais.
# Documentação do Matplotlib no link https://matplotlib.org/contents.html

x1 = cjdados.Date
y1 = cjdados.Close
data = [go.Scatter(x = x1, y = y1)]
layout = go.Layout(
    xaxis = dict(
        range = ['01-01-2010','11-04-2017'],
        title = 'Ano'              
    ),
    yaxis = dict(
        range = [min(x1), max(y1)],
        title = 'Valor da Ação'
    ))
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# In[ ]:


# Visualizando os Candlesticks através do gráfico.
# Podemos observar que os parâmetros são as colunas do meu conjunto de dados que estamos inserindo na variável dados.

cjdados2 = cjdados.head(7)
dados = go.Candlestick(x = cjdados2.Date,
                       open = cjdados2.Open,
                       high = cjdados2.High,
                       low = cjdados2.Low,
                       close = cjdados2.Close,
                       )

data = [dados]
py.offline.iplot(data,filename='grafico_candlestick')


# In[ ]:


# Visualizando precos em formato de Candlesticks dos últimos 6 meses.

cjdados2 = cjdados.head(180)
dados = go.Candlestick(x = cjdados2.Date,
                       open = cjdados2.Open,
                       high = cjdados2.High,
                       low = cjdados2.Low,
                       close = cjdados2.Close,
                       )

data = [dados]
py.offline.iplot(data,filename='grafico_candlestick')


# In[ ]:


# Plota a variação no período do conjunto de dados 2010 até 2017.
# O camando matplotlib é usado para exibir o gráfico no próprio jupyter notebook.

get_ipython().run_line_magic('matplotlib', 'notebook')

# Importação das bibliotecas que serão utilizadas.

import matplotlib.dates as mdates
import datetime as dt

# Dados para criação do gráfico.

x = cjdados['Date']
y = cjdados['Variation']
plt.plot_date(x,y, color='b',fmt="r-")
plt.xticks(rotation=30)
plt.show()


# In[ ]:


# Criação de uma variável treino que recebe todas as informações do nosso conjunto de dados.
# O objeto é deixar os dados anteriores intactos.
dadosTreino = cjdados


# In[ ]:


# Plota a dispersão entre o preço de abertura(Open) e fechamento(Close) dos últimos 100 dias.
# Para pegar os últimos dias, podemos observar a sintaxe :100 como parâmetro.

get_ipython().run_line_magic('matplotlib', 'notebook')

x = dadosTreino.Open[:100]
y = dadosTreino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('preco de abertura')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.show()


# In[ ]:


# Plota a dispersao entre o preço de máxima (high) e fechamento(Close) dos últimos 100 dias.

get_ipython().run_line_magic('matplotlib', 'notebook')
x = dadosTreino.High[:100]
y = dadosTreino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('preco da maxima')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.show()


# In[ ]:


# Plota a dispersao entre o preço de mínima(Low) e fechamento(Close) dos últimos 100 dias.

get_ipython().run_line_magic('matplotlib', 'notebook')
x = dadosTreino.Low[:100]
y = dadosTreino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('preco de Minima')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.show()


# In[ ]:


# Plota a dispersao entre o Volume e fechamento(Close) dos últimos 100 dias.

get_ipython().run_line_magic('matplotlib', 'notebook')
x = dadosTreino.Volume[:100]
y = dadosTreino.Close[:100]
plt.scatter(x,y,color='b')
plt.xlabel('Volume')
plt.ylabel('preco de fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.ticklabel_format(style='plain', axis='x')
plt.autoscale('False')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Vamos criar uma variável feature com os nomes das respectivas colunas.
# Criação de uma variável treino para receber as respectivas features.

features = ['Open','High','Low','Volume']
dadosTreino = dadosTreino[features]


# In[ ]:


# Visualizando os dados sem as classes. A nossa classe é a feature de fechamento (close)

dadosTreino.head()


# In[ ]:


# Criando a variável y para receber o preço de fechamento (classes)
y = cjdados['Close']


# In[ ]:


# Visualizando o dataframe y com as respectivas classes (fechamento)
y


# In[ ]:


# Realizando o treinamento do algoritmo de Regressão Linear
# Separando os dados de teste e de treino. 
# Usando o recurso **train_test_split** para separar dados de treino e teste.
# Dessa forma o algoritmo é treinado com uma parte dos dados e testado com outra (dados não vistos).
# Divisão dos dados de forma aleatória (75% para treino e 25% para teste).

X_treino, X_teste, y_treino, y_teste = train_test_split(dadosTreino, y, random_state = 42)


# In[ ]:


# Visualizando o dataframe X_treino
X_treino.head()


# In[ ]:


# Visualizando dados de teste.
X_teste.head()


# In[ ]:


# Visualizando as classes de treino.
y_treino.head()


# In[ ]:


# Visualizando as classes do teste.
y_teste.head()


# In[ ]:


# Cria um objeto do tipo LinearRegression.
modeloLr = LinearRegression()


# In[ ]:


# Treinando o algoritmo através do método fit.

modeloLr.fit(X_treino,y_treino)


# In[ ]:


# Visualizando os coeficientes (pesos!)
# Interessante observar o valor negativo do peso associado a feature Open (Abertura).

modeloLr.coef_


# In[ ]:


# Predizendo 10 preços com o método predict.

modeloLr.predict(X_teste)[:10]


# In[ ]:


# Visualizando preços reais.
y_teste[:10]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# Armazena dados preditos em dataframe.
predicoes = pd.DataFrame(modeloLr.predict(X_teste)[:10])

# Armazena dados reais em dataframe.
y_teste2= pd.DataFrame(y_teste[:10].values)

# Define o estilo do gráfico.

plt.style.use("ggplot")

# Definição de título de eixos do gráfico.

plt.xlabel('Preços')
plt.ylabel('Indice')
plt.title('Precos Reais vs Predições')

# Ordena os valores e plota as linhas
plt.plot(predicoes.sort_values(by=0),predicoes.index)
plt.plot(y_teste2.sort_values(by=0),y_teste2.index)

# Define legenda do gráfico
plt.legend(['Predições','Preços Reais'])


# In[ ]:


# Validando o modelo de Regressão Linear.
# Métricas de RMSE - utiliza medidas dependentes.

y_teste.isnull().sum()


# In[ ]:


y_pred = modeloLr.predict(X_teste)


# In[ ]:


y_pred.shape


# In[ ]:


y_teste.shape


# In[ ]:


# Utilizando o mean_squared_error. Significa o erro médio no conjunto de dados.
# Quanto mais próximo de 0 melhor.

mean_squared_error(y_teste, modeloLr.predict(X_teste))


# In[ ]:


# Vamos tentar melhorar os resultados do nosso modelo.
# Vamos usar apenas as duas features.
# Criando uma nova variável para o novo modelo de predição.

modeloLr2 = LinearRegression(normalize=True)


# In[ ]:


# A variável features está recebendo apenas o valor de abertura e o valor máximo.

features = ['Open','High']
dadosTreino2 = dadosTreino[features]


# In[ ]:


# Visualizando os dados atráves do método head().
# Exibe apenas os 05 primeiros registros.

dadosTreino2.head()


# In[ ]:


# Separa os dados 75% treino e 25% teste
# Essa etapa já foi executada anteriormente.

X_treino, X_teste, y_treino, y_teste = train_test_split(dadosTreino2, y, random_state=42)


# In[ ]:


# Treinando o algoritmo com apenas duas features.

modeloLr2.fit(X_treino,y_treino)


# In[ ]:


# Imprimi os pesos dos coeficientes.
modeloLr2.coef_


# In[ ]:


# Valida o modelo com o RMSE.
RMSE = mean_squared_error(y_teste, modeloLr2.predict(X_teste))**0.5
RMSE

