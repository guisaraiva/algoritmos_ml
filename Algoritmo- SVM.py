
# coding: utf-8

# In[ ]:


# Bibliotecas utilizadas no algoritmo SVM
# sklearn.model_selection (utilizamos para dividir o dataset em treino e teste posteriormente.)
# Matplotlib (biblioteca gráfica)
# Uma ótima leitura na documentação do Sklearn (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)


from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importação do dataset IRIS. Esse conjunto de dados é simples e de fácil manipulação.
# Irei começar a executação do algoritmo nessa base de dados e posteriormente utilizei um dataset spotfy.

from sklearn import datasets


# In[ ]:


# Carregando o dataset para o array chamado iris
# A partir do sklearn podemos carregar o dataset sem a necessidade de importação de arquivo externo.
# Em outros algoritmos, realizei a importação do dataset através de um arquivo.

iris = datasets.load_iris()


# In[ ]:


# Um método que já aprendi anteriormente.
# O type() retorna o tipo de uma variável criada anteriormente.

type(iris)


# In[ ]:


# Visualizando o nome das features
# Podemos chamar vários métodos no objeto iris.
# Lembrando que estou usando os métodos pois não carregamos os dados de arquivo externo.
# Coisas novas por causa do sklearn.

iris.feature_names


# In[ ]:


# Nome das Classes
# Repare que é um array. Podemos identificar pelo colchete []

iris.target_names


# In[ ]:


# Separando dados de treino.
# Armazenando os dados do dataset em uma variável treino.

treino = iris.data
#treino


# In[ ]:


# Separando dados de classes. 
# Criei um variável chamada classe. Poderia ser qualquer nome.

classe = iris.target
#classe


# In[ ]:


# Visualizando a forma do array dos dados de treino. Formato (linhas,colunas).
# O resultado é organizado conforme explicado anteriormente. 150 linhas com 4 colunas.

treino.shape


# In[ ]:


# Visualizando os dados de treino.
# Passei o parâmetro :20. Vai exibir os 20 últimos registros do dataset de treino.

treino[:20]


# In[ ]:


# Visualizando a forma do array de classes.
# Na classe, vamos observar apenas as informações de 150 registros.

classe.shape
#classe


# In[ ]:


# Visualizando os dados únicos do array de classes.
# O set executa a mesma ação que um comando distinct (oracle) !! Pra lembrar melhor !! :)
set(classe)


# In[ ]:


# Visualizando os dados de classes.

classe[:]


# In[ ]:


# Visualização de dados. Visualizando a disperssão de dados quanto a classe
# Agora posso usar a biblioteca matplotlib para plotar os gráficos com as respectivas features.
# Plota gráfico de disperssão dos dados com relação a classe.
# Disperssão dos dados de Sepal width e Sepal Length com as classes(0,1,2)
# Documentação para ajudar a entender melhor (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html)

get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("ggplot") # O stilo definido para o gráfico.
#sepal length vs sepal width
plt.xlabel('Sepal length') # A descrição do eixo X
plt.ylabel('Sepal width') # A descrição do eixo Y
plt.title('Sepal width vs Sepal length') # O título do gráfico.
plt.scatter(treino[:,0],treino[:,1], c=classe)


# In[ ]:


# Plota gráfico de disperssão dos dados com relação a classe.
# Disperssão dos dados de Petal width e Petal Length com as classes(0,1,2)

get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("ggplot")
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Petal Width vs Petal Length')
plt.scatter(treino[:,2], treino[:,3], c=classe)


# In[ ]:


# Realizar a aplicação do algoritmo SVM para classificar as flores usando o dataset IRIS.
# Pegar 80% dos dados para treino e 20% para teste (Normalmente pegamos essa quantidade).

# Visualizando o tamanho dos dados de treino.
# O parâmetro -30 é para retirar os apenas os dados necessários para treino.
# Os demais serão utilizados para teste.

len(treino[:-30]), len(classe[:-30])


# In[ ]:


# Visualizando os dados de treino

treino[:-30]


# In[ ]:


# Visualiando as classes correspondentes
# A classe desse bloco de execução pertence aos dados de treino.

classe[:-30]


# In[ ]:


# Treinando o algoritmo de SVM. Agora começa a brincadeira !!! ;)
# O método fit sendo invocado para treinar os dados.


classificadorSVM = svm.SVC().fit(treino[:-30],classe[:-30])


# In[ ]:


# Objeto classificadorSVM
# Analisando o resultado podemos observar vários parâmetros que existem no SVC.
# Vou deixar todos parâmetros com os valores padrões.

classificadorSVM


# In[ ]:


# Cria um array com os dados de teste.
# 20% dos dados que não foram testados.
# Lembrando que separei os 80% do dataset anteriormente.

teste = treino[-30:]
teste


# In[ ]:


# Predizendo valores com a porção de dados de teste.
# Chamamos o objeto criado com o método predict do SVN.
# Passamos os dados de teste 20%

classificadorSVM.predict(teste)


# In[ ]:


# Cria um array com as classes dos dados de teste.

classe_teste = classe[-30:]
classe_teste


# In[ ]:


# Visualizando os Resultados de Classificação.
# Gráfico de disperssão entre as colunas Sepal Length , Sepatl width e a classe.
# Os pontos roxos são pontos no qual o classificador errou.

get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("ggplot")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Sepal width vs Sepal length')
plt.scatter(treino[-30:,0],treino[-30:,1], c=classificadorSVM.predict(teste))


# In[ ]:


# Gráfico de disperssão entre as colunas Petal Length , Petal width e a classe.

get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("ggplot")
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Petal Width vs Length')
plt.scatter(treino[-30:,2], treino[-30:,3], c=classificadorSVM.predict(teste))


# In[ ]:


# Gráfico de instâncias e predição destas.

get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("ggplot")
plt.xlabel('Amostras')
plt.ylabel('Classes')
plt.title('Classificacao do SVM')
plt.scatter(range(len(classe_teste)),classe_teste,c=classificadorSVM.predict(teste))


# In[ ]:


# Utilizando a matriz de confusão.
# Link com referência massa: https://minerandodados.com.br/matriz-de-confusao/
# No resultado da matriz, o algoritmo preveu um 2 como 1 em cinco momentos.

print (pd.crosstab(classe_teste,classificadorSVM.predict(teste),rownames=['Real'], colnames=['Predito'], margins=True),'')


# In[ ]:


# Usando o Cross Validation.
# Link com uma explicação sobre Cross Validation 
# (https://medium.com/data-hackers/como-criar-k-fold-cross-validation-na-m%C3%A3o-em-python-c0bb06074b6b)

# Criei uma função que retorna a acurácia após fazer um validação cruzada (cross validation)


def Acuracia(classificadorSVM,X,y):
    resultados = cross_val_predict(classificadorSVM, X, y, cv=10)
    return metrics.accuracy_score(y,resultados)


# In[ ]:


# Chamado a função criada anteriormente e passando os parâmetros necessários para análise.
# Uma acurácia de 98% é boa.... 

Acuracia(classificadorSVM,treino,classe)


# In[ ]:


# Utilizar métricar de avaliação.
# Imprime as métricas: 'precisão, revocação e Medida F1.
# Link massa para absorção de conhecimento:
# https://paulovasconcellos.com.br/como-saber-se-seu-modelo-de-machine-learning-est%C3%A1-funcionando-mesmo-a5892f6468b

# Armazenando o resultado do cross validation na variável resultados.
resultados = cross_val_predict(classificadorSVM,treino, classe, cv=10)

# Criando valores fixos
valor_classes = [0,1,2]

# Exibindo as métricas de avaliação através do classifcation_report do pacote sklearn
print (metrics.classification_report(classe,resultados,valor_classes))


# In[ ]:


# Informações das Features da Base de dados.
# https://developer.spotify.com/web-api/get-audio-features/
# Vou aplicar o algoritmo em uma outra base de dados. Uma base de dados do Spotfy para realizarmos a aplicação do SVM.
# Vamos ver no que vai dá !!!
dataset = pd.read_csv('data.csv', sep=',')
dataset.head()


# In[ ]:


# Analisando os respectivos dados. Checando valores missing.
# O termo missing é muito utilizado em análise de dados. Valores ausentes ...
# Estou pedindo para exibir o total de valores nulos que existem no dataset.

dataset.isnull().sum()


# In[ ]:


# Resumo Estatístico da base. Somente dados numéricos.
# O describe é um excelente método estatístico.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

dataset.describe()


# In[ ]:


# lista estilos disponíveis do Matplotlib.
# Um método interessante para verificar quais estilos de gráficos estão disponíveis no Matplotlib.
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html

plt.style.available


# In[ ]:


# Vou plotar vários gráficos comparando as features do dataset.
# A comparação está ocorrendo com features analisadas anterioremente.

get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("seaborn-colorblind")
dataset.plot(x='acousticness', y='danceability', c='target', kind='scatter', colormap='Accent_r')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("seaborn-colorblind")
dataset.plot(x='tempo', y='valence', c='target', kind='scatter' , colormap='Accent_r')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("seaborn-colorblind")
dataset.plot(x='tempo', y='speechiness', c='target', kind='scatter' , colormap='Accent')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import style
style.use("seaborn-colorblind")
dataset.plot(x='danceability', y='energy', c='target', kind='scatter' , colormap='Reds')


# In[ ]:


# Começando a realizar a separação do dataset em treino e teste.
# Estou armazenando a variável target e posteriormente realizando a exclusão dela do dataset.

classesSpotfy = dataset['target']
dataset.drop('target', axis=1, inplace=True) 


# In[ ]:


# Visualizando os 15 primeiros registros do daataset
dataset.head(15)


# In[ ]:


# Realizar um pré-processamento dos dados. 
# A função remove_features executa a remoção de features que serão passadas como parâmetro posteriormente.
# Estou apenas criando a função neste momento.


def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0


# In[ ]:


# Remove features. Observe que estou passando duas features através de um array.
# O retorno é 0.

remove_features(['id','song_title'])


# In[ ]:


# Visualizando o dataset após aplicação dos comandos anteriores.

dataset.artist.head(10)


# In[ ]:


# Visualizando o dataset com o método info()
dataset.info()


# In[ ]:


# Vou sar o Label Encoder. Essa técnica codifica valores categóricos em numéricos.
# Realizando a importação do Label Encoder.

from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Instânciando o labelEncoder
lEncoder = LabelEncoder()


# In[ ]:


# Aplicando o fit_transform do labelEncoder. 
# Transformando a feature "artist" e dados numéricos.

inteiro = lEncoder.fit_transform(dataset['artist'])


# In[ ]:


# Visualizando valores únicos. Lembrar do distinct (oracle)

set(inteiro)


# In[ ]:


# Estou criando uma nova colun "artist_inteiro" que recebe os dados transformados.

dataset['artist_inteiro'] = inteiro
dataset.head()


# In[ ]:


remove_features(['artist'])


# In[ ]:


# Visualizando o Dataset alterado.
dataset.head(20)


# In[ ]:


# Importa o pacote OneHotEncoder
# Técnica usada para codificar valores categóricos em númericos.
# Resolve o problema __ordenação__ nos dados gerados pelo LabelEncoder.

from sklearn.preprocessing import OneHotEncoder


# In[ ]:


# Instancia um objeto do tipo OnehotEncoder

oHE = OneHotEncoder()
dataset.values


# In[ ]:


# Transforma em array numpy o dataset.

dataset_array = dataset.values


# In[ ]:


# Pega o numero de linhas.
num_rows = dataset_array.shape[0]

# Visualiza coluna de inteiros
dataset_array[:][:,13]


# In[ ]:


# Transforma a matriz em uma dimensão

inteiro = inteiro.reshape(len(inteiro),1)


# In[ ]:


# Criar as novas features a partir da matriz de presença.
novas_features = oHE.fit_transform(inteiro)

# Imprime as novas features
novas_features


# In[ ]:


# Concatena as novas features ao array
dataset_array = np.concatenate([dataset_array, novas_features.toarray()], axis=1)

# Visualizando a quantidade de linhas e colunas da base
dataset_array.shape


# In[ ]:


# Transforma em dataframe e visualiza as colunas
dataf = pd.DataFrame(dataset_array)
dataf.head(100)


# In[ ]:


# Aplicando o get_dummies nos dados.
# Cria uma matriz de presença como feito com o OHE.
# Vamos importar novamente o dataset.

#dataset = pd.read_csv('data.csv', sep=',')
dataset = pd.get_dummies(dataset, columns=['artist'], prefix=['artist'])


# In[ ]:


# Visualizando 'features' geradas.

dataset.columns


# In[ ]:


# Verificando o tamanho.

len(dataset.columns)


# In[ ]:


# Visualizando as colunas
dataset.dtypes


# In[ ]:


# checando missing values
dataset.isnull().sum()


# In[ ]:


# coluna artist
len(dataset.columns)


# In[ ]:


# Utilizando o Pipeline
# Importe as bibliotecas de Pipelines e Pré-processadores (normalização - standartization.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


classesSpotfy.head()


# In[ ]:


# Treinando o algoritmo de SVM.
classifSVM = svm.SVC().fit(dataset,classesSpotfy)


# In[ ]:


# Chamando novamente a função criada anteriormente.

Acuracia(classifSVM,dataset,classesSpotfy)


# In[ ]:


# Criando um pipeline 
# Podemos observar que estamos normalizando os dados com o pipeline.

pip_1 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC())
])


# In[ ]:


# Imprimindo Etapas do Pipeline
pip_1.steps


# In[ ]:


# Chama a função acuracia passando os dados de musicas e as classes
# Usando o pipeline pip_1

Acuracia(pip_1,dataset,classesSpotfy)


# In[ ]:


# Criando vários Pipelines
pip_2 = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('clf', svm.SVC())
])

pip_3 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='rbf'))
])

pip_4 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='poly'))
])

pip_5 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='linear'))
])


# In[ ]:


# Chama a função acuracia passando os dados de musicas e as classes
# Usando o pipeline pip_2
Acuracia(pip_2,dataset,classesSpotfy)


# In[ ]:


# Teste com apenas labelEncoder nos dados
# Teste com apenas LabelEncoder na coluna 'artist' usando o pipeline 'pip_1'

Acuracia(pip_1,dataset,classesSpotfy)


# In[ ]:


# # Teste com apenas LabelEncoder na coluna 'artist' usando o pipeline 'pip_2'

Acuracia(pip_2,dataset,classesSpotfy)


# In[ ]:


# Testando o desempenho dos kernels.
# Testando o Kernel RBF

Acuracia(pip_3,dataset,classesSpotfy)


# In[ ]:


# Teste de kernel poly
Acuracia(pip_4,dataset,classesSpotfy)


# In[ ]:


# Teste de Kernel linear
Acuracia(pip_5,dataset,classesSpotfy)


# In[ ]:


# Realizando o tunning nos dados. 
# Importa o utilitário GridSearchCV
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Lista de Valores de C
lista_C = [0.001, 0.01, 0.1, 1, 10, 100]

# Lista de Valores de gamma
lista_gamma = [0.001, 0.01, 0.1, 1, 10, 100]


# In[ ]:


# Define um dicionário que recebe as listas de parâmetros e valores.

parametros_grid = dict(clf__C=lista_C, clf__gamma=lista_gamma)
parametros_grid


# In[ ]:


# Objeto Grid recebe parâmetros de Pipeline, e configurações de cross validation
grid = GridSearchCV(pip_3, parametros_grid, cv=10, scoring='accuracy')

# Aplica o gridsearch passando os dados de treino e classes.
grid.fit(dataset,classesSpotfy)


# In[ ]:


# Resultados de Grid
# Imprime os scores por combinações
grid.grid_scores_


# In[ ]:


# Imprime os melhores parâmetros
grid.best_params_


# In[ ]:


grid.best_score_

