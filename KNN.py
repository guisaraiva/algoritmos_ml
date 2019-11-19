
# coding: utf-8

# In[ ]:


# Importação das bibliotecas que serão utilizadas na aprendizagem do algoritmo KNN.
# A novidade está relacionada a biblioteca Warnings. Ela irá inibir uso de tecnologia obsoleta.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Carregando a famosa base de dados Iris. Contém os dados de classificação de tipo de flores.
# Já poderíamos importar a base de dados através da biblioteca Scikit Learn datasets.

iris = pd.read_csv("iris.csv")


# In[ ]:


#  Verificando  os atributos e features do conjunto de dados.

iris.head()


# In[ ]:


# Usando o método infor() para retornar informações do conjunto de dados.

iris.info()


# In[ ]:


# Usando o método describe() para retornar informações estatísticas.

iris.describe()


# In[ ]:


# Realizando a divisão dos dados de treino e teste através do train_test_split do Scitkit Learn.
# Estamos dropando a coluna Species com a classificação e armazenando em outra variável tais informações.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.drop('Species',axis=1),iris['Species'],test_size=0.3)


# In[ ]:


# Verificando a forma dos dados após a divisão dos dados (teste e treino)

X_train.shape,X_test.shape


# In[ ]:


# Verificando a forma dos dados após a divisão dos dados (teste e treino)

y_train.shape,y_test.shape


# In[ ]:


# Instânciando o algoritmo KNN passando o parâmetro de 3 vizinhos.

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


# Treinando o algoritmo através do método fit.

knn.fit(X_train,y_train)


# In[ ]:


# Executando o KNN com o conjunto de teste.

resultado = knn.predict(X_test)
resultado


# In[ ]:


# Criando novas amostrar para testar o modelo criado anteriormente.
# Uma variável test está sendo criada e recebendo valores fixos no array;

test = np.array([[5.1,3.5,1.4,0.2]])
knn.predict(test),knn.predict_proba(test)


# In[ ]:


# Executando a Matriz de confusão para verificar o resultado da predição e dos dados reais.

print (pd.crosstab(y_test,resultado, rownames=['Real'], colnames=['Predito'], margins=True))


# In[ ]:


# Executando métricas de classificações através do metrics do ScikitLearn.
# Documentação do metrics:  https://scikit-learn.org/stable/modules/classes.html
from sklearn import metrics
print(metrics.classification_report(y_test,resultado,target_names=iris['Species'].unique()))


# In[ ]:


# Importando o conjunto de dados dígitos.
# Classificando e performando o metrics do Scikit Learn.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

# Conjunto de dados dígitos.

digitos = datasets.load_digits()


# In[ ]:


# DEscrição sobre a bade de dados carregada anteriormente na variável digits.

print(digitos.DESCR)


# In[ ]:


# Visualizando os valores dos dados exisntes no conjunto de dados.

digitos.images


# In[ ]:


# Visualizando os valores de clases do conjunto de dados.

digitos.target_names


# In[ ]:


# Visualizando as imagens e classes do conjnto de dados.

images_and_labels = list(zip(digitos.images, digitos.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


# In[ ]:


# Convertendo os dados em um Dataframe. 

n_samples = len(digitos.images)
data = digitos.images.reshape((n_samples, -1))
classe = digitos.target


# In[ ]:


dataset = pd.DataFrame(data)
dataset['classe'] = classe


# In[ ]:


dataset.head()


# In[ ]:


# Realizando a divisão dos dados de treino e teste. Da mesma forma que foi realizado anteriomente.
# A variável target é a classe do conjunto de dados.

X_train, X_test, y_train, y_test = train_test_split(dataset.drop('classe',axis=1),dataset['classe'],test_size=0.3)


# In[ ]:


# Visualização a forma dos dados.

X_train.shape,X_test.shape


# In[ ]:


# Visualização a forma dos dados.

y_train.shape,y_test.shape


# In[ ]:


# Instânciando o algoritomo KNN

knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


# Treinando o algoritmo criado através do método fit.

knn.fit(X_train,y_train)


# In[ ]:


# Predizendo novos pontos para nosso modelo.

resultado = knn.predict(X_test)


# In[ ]:


# Técnica de métricas de classificação.

from sklearn import metrics
print(metrics.classification_report(y_test,resultado))


# In[ ]:


# Matriz de confusão para validar as acertabilidades.

print (pd.crosstab(y_test,resultado, rownames=['Real'], colnames=['Predito'], margins=True))


# In[ ]:


# Aplicação do Cross Validadtion 

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, dataset.drop('classe',axis=1),dataset['classe'], cv=5)
scores


# In[ ]:


# Realizando um tunning no parâmetro K
# Importando o GridSearchCV

from sklearn.model_selection import GridSearchCV


# In[ ]:


# Criando um range de valores para realização do tunning do modelo.

k_list = list(range(1,31))


# In[ ]:


# Criando um dicionário e atribuindo os dados do range no n_neighbors (parâmetro) do GridSearchCV

k_values = dict(n_neighbors=k_list)
k_values


# In[ ]:


# Instânciando o objeto GridSearchCV na variável grid.

grid = GridSearchCV(knn, k_values, cv=5, scoring='accuracy')


# In[ ]:


# Treinando o algoritmo novamente através do método .fit

grid.fit(dataset.drop('classe',axis=1),dataset['classe'])


# In[ ]:


# Visualizando os valores de scores do algoritmo.

grid.best_score_


# In[ ]:


# Exibindo os respectivos valores de K com a sua acurácia.
# Retorna o melhor valor de K.

print("Melhor valor de k = {} com o valor {} de acurácia".format(grid.best_params_,grid.best_score_))


# In[ ]:


# Visualizando os valores de K e acurácia em forma de gráfico.

get_ipython().run_line_magic('matplotlib', 'notebook')

scores=[]
results = grid.cv_results_
for mean in results['mean_test_score']: scores.append(mean)

plt.figure(figsize=(10,6))
plt.plot(k_list,scores,color='red',linestyle='dashed',marker='o')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()

