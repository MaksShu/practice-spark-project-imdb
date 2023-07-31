import pyspark
from columns import *

from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.types as t
import pyspark.sql.functions as f

# Initialize Spark Session
spark_session = (SparkSession.builder
                             .master("local")
                             .appName("task app")
                             .config(conf=SparkConf())
                             .getOrCreate())

# Set paths to datasets files
names_path = './data/name.basics.tsv.gz'
basics_path = './data/title.basics.tsv.gz'
crew_path = './data/title.crew.tsv.gz'
ratings_path = './data/title.ratings.tsv.gz'


# Read datasets into dataframes
def create_dataframes():
    # Create schema for names dataset
    names_schema = t.StructType([t.StructField(name_id, t.StringType(), False),
                                 t.StructField(name_primary_name, t.StringType(), False),
                                 t.StructField(name_birth_year, t.IntegerType(), True),
                                 t.StructField(name_death_year, t.IntegerType(), True),
                                 t.StructField(name_primary_profession, t.StringType(), True),
                                 t.StructField(name_known_for, t.StringType(), True)])

    # Create names dataframe from csv file, where separator is \t, columns has headers and null value is \N
    names_df = spark_session.read.csv(names_path, schema=names_schema, sep='\t', header=True, nullValue="\\N")

    # Create schema for basics dataset
    basics_schema = t.StructType([t.StructField(title_id, t.StringType(), False),
                                  t.StructField(title_type, t.StringType(), True),
                                  t.StructField(title_primary, t.StringType(), True),
                                  t.StructField(title_original, t.StringType(), True),
                                  t.StructField(title_adult, t.IntegerType(), True),
                                  t.StructField(title_start_year, t.IntegerType(), True),
                                  t.StructField(title_end_year, t.IntegerType(), True),
                                  t.StructField(title_minutes, t.StringType(), True),
                                  t.StructField(title_genres, t.StringType(), True)])

    # Create basics dataframe from csv file, where separator is \t, columns has headers and null value is \N
    basics_df = spark_session.read.csv(basics_path, schema=basics_schema, sep='\t', header=True, nullValue="\\N")

    # Create schema for crew dataset
    crew_schema = t.StructType([t.StructField(title_id, t.StringType(), False),
                                t.StructField(crew_directors, t.StringType(), True),
                                t.StructField(crew_writers, t.StringType(), True)])

    # Create crew dataframe from csv file, where separator is \t, columns has headers and null value is \N
    crew_df = spark_session.read.csv(crew_path, schema=crew_schema, sep='\t', header=True, nullValue="\\N")

    # Create schema for ratings dataset
    ratings_schema = t.StructType([t.StructField(title_id, t.StringType(), False),
                                   t.StructField(ratings_average, t.FloatType(), True),
                                   t.StructField(ratings_votes, t.IntegerType(), True)])

    # Create ratings dataframe from csv file, where separator is \t, columns has headers and null value is \N
    ratings_df = spark_session.read.csv(ratings_path, schema=ratings_schema, sep='\t', header=True, nullValue="\\N")

    return names_df, basics_df, crew_df, ratings_df


names_df, basics_df, crew_df, ratings_df = create_dataframes()

names_df.show()
names_df.printSchema()
names_df.describe().show()

basics_df.show()
basics_df.printSchema()
basics_df.describe().show()

crew_df.show()
crew_df.printSchema()
crew_df.describe().show()

ratings_df.show()
ratings_df.printSchema()
ratings_df.describe().show()
