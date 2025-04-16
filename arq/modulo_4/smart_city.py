from pyspark.ml.classification import LogisticRegression  # utilizada para realizar a classifcação
from pyspark.ml.feature import RegexTokenizer  # utilizada para realizar a tokenização (divisão da senteça em palavras)
from pyspark.ml.feature import \
    StopWordsRemover  # utilizada para remover as stopwords (palavras "sem sentido" para a análise)
from pyspark.ml.feature import CountVectorizer  # transforma as palavras em vetores
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler  # utilizadsa para transformar os dados
from pyspark.ml import Pipeline  # utilizada para criar o pipeline de transformações
from pyspark.sql.functions import col  # utilizada para selecionar as colunas
from pyspark.sql import SparkSession  # utilizada para iniciar a seção do spark
from pyspark.ml.evaluation import \
    MulticlassClassificationEvaluator  # utilizado para realizar a avaliação de classificadores não binários
from pyspark.ml.classification import LogisticRegression  # utilizada para realizar a classifcação

spark = SparkSession.builder.appName("PrevisaoDeCrimes").getOrCreate()  # inicia a seção do spark
# %%
# Leitura dos dados
dir_data = '/home/gelson/datasets/Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
data = spark.read.format("csv").options(header="true", inferschema="true").load(dir_data)
data.columns
data.count()
data.printSchema()  # realiza o print do schema para o bando de dados lido
data.show(10)
drop_data = ['IncidntNum', 'Date', 'Time', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'PdId',
             'Location']  # indica quais as colunas vamos eliminar (vamos ficar apenas com "Category" e "Descript")
# data = data.drop(*drop_data)  #realiza a exclusão das colunas indicadas
# aplica a seleção apenas da colunas que não estão na lista "drop_data"
data = data.select([column for column in data.columns if column not in drop_data])
data.show(10, False)
data.describe().show()
data.groupby('Category').count().show()
exit(0)
# aplicando expressões regulares (regular expression)
re_Tokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")

# removendo stopwords
# cria o vetor que contém as palavras que não geram sentido ao texto
stop_words = ["http", "https", "amp", "rt", "t", "c", "the"]
# aplica a remoção das stopwords contidas no vetor
stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stop_words)
# Transformando as palavras em vetores (BagOfWords)
count_vectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
# aplicando a transformação (string -> numerica) de cada uma das categorias
label_string_Idx = StringIndexer(inputCol="Category", outputCol="label")
# define todas as operação através do pipeline
# diz a sequência das transformações a serem realizadas
pipeline = Pipeline(stages=[re_Tokenizer, stop_words_remover, count_vectors, label_string_Idx])
# Realmente, aplica as transformações
pipeline_fit = pipeline.fit(data)  # aplica as transformações
newDataset = pipeline_fit.transform(data)
newDataset.show(10, False)
(trainingData, testData) = newDataset.randomSplit([0.7, 0.3], seed=100)
print("Comprimento do Treinamento: " + str(trainingData.count()))
print("Comprimento do Teste: " + str(testData.count()))
# Constrói o modelo de classificação
regressor = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
# aplica a regressão aos dados de treinamento
modeloRegressaoL = regressor.fit(trainingData)
# realiza a previsão utilizando o modelo de regressão logística
previsao = modeloRegressaoL.transform(testData)
# avalia o modelo de previsão construído
# funcao utilizada para realizar a avaliação de classifcadores com várias classes
avaliacao = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='f1')
#avaliacao.evaluate(previsao)  # accuracy, areaUnderROC, accuracy
print(avaliacao.evaluate(previsao))