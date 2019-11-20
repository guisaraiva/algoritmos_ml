
# coding: utf-8

# In[ ]:


# Importando as bibliotecas necessárias para utilização do algoritmo K-Means

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[ ]:


# Carregando a base de dados Iris para execução do algoritmo.

iris = pd.read_csv("iris.csv")


# In[ ]:


# Verificando os cinco primeiros registros do dataset

iris.head()


# In[ ]:


# Plotando a imagem "exemplo" que demonstra as features existentes na base de dados.

from IPython.display import Image
Image(filename ="iris-data-set.png", width=500, height=500)


# In[ ]:


# Separando os dados e classes do conjunto de dados iris.
# Dropei a coluna "Species" que armazena a variável target do nosso treinamento.

X = iris.drop('Species',axis=1)
X[:10]


# In[ ]:


# Armazenando a variável target em outra variável para que possamos utilizar depois.
# O método unique() exibe apenas as informações existentes na feature Species. 
# É como se estivesse executando um distinct no banco de dados.

y = iris.Species
y.unique()


# In[ ]:


# Função criada para converter os valores categóricos da classe em números. 
# O algoritmo é utilizando em variáveis numéricas.
# A função é composta de três condições if

def converte_classe(l):
    if l == 'Iris-virginica':
        return 0
    elif l == 'Iris-setosa':
        return 1
    elif l == 'Iris-versicolor':
        return 2


# In[ ]:


# A variável y recebe o conjunto de features com o método apply(), retornando a função no parâmetro.
# O método count para contar os respectivos valores

y = y.apply(converte_classe)
y.value_counts()


# In[ ]:


# Instânciando o algoritmo K-Means com 3 clusters.
# A documentação ajuda a compreender melhor.
# Link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

kmeans = KMeans(n_clusters = 3, init = 'random')


# In[ ]:


# Treinando o algoritmo com o método .fit.

kmeans.fit(X)


# In[ ]:


# Verificando os centroids através do cluster_centers.
# Estes centroids são pontos de dados que serão utilizados, como o nome sugere, de pontos centrais dos clusters.
# Maiores detalhes no blog Mineirando dados.
# Link: https://minerandodados.com.br/entenda-o-algoritmo-k-means/

kmeans.cluster_centers_


# In[ ]:


# Criando a variável distância com o método fit_transform

distance = kmeans.fit_transform(X)
distance


# In[ ]:


distance[0]


# In[ ]:


# Visualizando valores de distância para cada cluster.
# Usando o matplotlib para plotar o gráfico.
# Gráfico do tipo barras horizontais.

get_ipython().run_line_magic('matplotlib', 'notebook')
x = ['Cluster 0','Cluster 1','Cluster 2']
plt.barh(x,distance[0])
plt.xlabel('Distância')
plt.title('Distância por Clusters ')
plt.show()


# In[ ]:


# Imprimindo os respectivos labels.
# Podemos observar que ele transformou cada "Specie" em um valor numérico.

labels = kmeans.labels_
labels


# In[ ]:


# Visualizando os centroids através do gráfico.
# Os centroids são os pontos em vermelho.

get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure(figsize=(8,6))
plt.scatter(X['SepalLength'], X['SepalWidth'], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.title('Dataset Iris e Centroids')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.show()


# In[ ]:


# Criando um agrupamento com dados aleatórios. 
# Chamamos o método predict passando como parâmetro os dados aleatórios.

data = [
        [ 4.12, 3.4, 1.6, 0.7],
        [ 5.2, 5.8, 5.2, 6.7],
        [ 3.1, 3.5, 3.3, 3.0]
    ]
kmeans.predict(data)


# In[ ]:


# Visualizando os respectivos resultados através do gráfico.
# Podemos observar que o gráfico exibe os resultados originais e o resultado do algoritmo.

get_ipython().run_line_magic('matplotlib', 'notebook')
f,(ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,6))
ax1.set_title('Original')
ax1.scatter(X['SepalLength'], X['SepalWidth'],s=150,c=sorted(y))
ax2.set_title('KMeans')
ax2.scatter(X['SepalLength'], X['SepalWidth'],s=150,c=sorted(kmeans.labels_))


# In[ ]:


# Utilizamos o método Elbow para estimar o valor do parâmetro K.
# Um for para criar um range de valores.

get_ipython().run_line_magic('matplotlib', 'notebook')
wcss = []

for i in range(1, 11):
    kmeans2 = KMeans(n_clusters = i, init = 'random')
    kmeans2.fit(X)
    print (i,kmeans2.inertia_)
    wcss.append(kmeans2.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') 
plt.show()


# In[ ]:


# Verificando as informações com a Matriz de confusão.

print (pd.crosstab(y,kmeans.labels_, rownames=['Real'], colnames=['Predito'], margins=True))


# In[ ]:


# Utilizamos uma métrica de classificação.
# O pacote metrics do ScikitLearn.

from sklearn import metrics
clusters = ['Cluster 2','Cluster 1','Cluster 0']
print(metrics.classification_report(y,kmeans.labels_,target_names=clusters))

