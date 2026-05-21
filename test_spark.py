import os
import sys
from pyspark.sql import SparkSession

# Forzar a Spark a utilizar el ejecutable de Python de tu entorno virtual activo
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 1. Crear la sesión de Spark
spark = SparkSession.builder \
    .appName("PruebaSpark") \
    .master("local[*]") \
    .getOrCreate()

# 2. Crear unos datos de prueba
data = [("Juan", 25), ("María", 30), ("Pedro", 28)]
columns = ["Nombre", "Edad"]

# 3. Crear un DataFrame
df = spark.createDataFrame(data, schema=columns)

# 4. Mostrar los resultados en consola
df.show()

# 5. Detener la sesión
spark.stop()