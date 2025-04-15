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

wine_df = string_index.fit(wine_df).transform(wine_df)
