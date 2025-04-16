from pyspark.ml.classification import LogisticRegression as LR
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
# importa a função para criar uma nova coluna no dataset (similar às funções do SQL)
from pyspark.sql.functions import when

# inicia a seção para a utilização do spark
spark = SparkSession.builder.appName("ClassificadorVinhos").getOrCreate()

wine_df = spark.read.format("csv").options(header="true", inferschema="true", sep=';').load('winequality-red.csv')
wine_df.printSchema()
wine_df.show(10)

# realizando um filtro para o dataframe
wine_df.filter(wine_df.quality > 7).show(5)
wine_df.filter(wine_df.quality > 8).show(5)

wine_df = wine_df.withColumn('quality_new',
                             when(wine_df['quality'] < 5, 0).otherwise(
                                 when(wine_df['quality'] < 8, 1).otherwise(2)))
wine_df.show(15)

wine_df.filter(wine_df.quality_new > 1).show(5)

# converte os valores para string - não executa, é apenas transformação
print('converte os valores para string - não executa, é apenas transformação')
string_index = StringIndexer(inputCol='quality_new', outputCol='quality' + 'Index')

# wine_df = string_index.fit(wine_df).transform(wine_df)
# Seleciona as colunas a serem utilizadas como entradas para a classificação - não executa, é apenas transformação
vectors = VectorAssembler(
    inputCols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'], outputCol='features')
# cria o "novo dataframe" - não executa, é apenas transformação
stages = [vectors, string_index]
pipeline = Pipeline().setStages(stages)
# aplicando as transformações ao dataset - executa o comando/ação
pipelineModel = pipeline.fit(wine_df)
# aplica as transformações obtidas através do pipeline
pl_data_df = pipelineModel.transform(wine_df)
# print do dataset após a aplicação do pipeline
pl_data_df.show(15)
# divide o dataset entre treinamento e teste (70% treinamento e 30% teste)
train_df, test_df = pl_data_df.randomSplit([0.7, 0.3])
train_df.show(5)
test_df.show(5)
# instancia a classe para a execução do modelo através da regressão logística
classificador = LR(featuresCol='features', labelCol='qualityIndex', maxIter=50)  # maximo de iterações
# aplica o treinamento do modelo
modelo = classificador.fit(train_df)
# obtém o sumário (dados de análise) para o modelo criado
modelSummary = modelo.summary
# print das estatísticas do modelo
print(modelSummary)
# print das estatísticas do modelo
accuracy = modelSummary.accuracy  # acurácia da classificação
fPR = modelSummary.weightedFalsePositiveRate  # taxa de falsos positivos
tPR = modelSummary.weightedTruePositiveRate  # taxa de verdadeiros positivos
fMeasure = modelSummary.weightedFMeasure()  # f-score
precision = modelSummary.weightedPrecision  # precision
recall = modelSummary.weightedRecall  # recall
print("Acurácia: {} Taxa de verdadeiros positivos {} F-score {} Precision {} Recall {}".format(accuracy, tPR, fMeasure,
                                                                                              precision, recall))
