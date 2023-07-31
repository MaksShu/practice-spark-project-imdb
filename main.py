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
titles_path = './data/title.basics.tsv.gz'
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
    titles_schema = t.StructType([t.StructField(title_id, t.StringType(), False),
                                  t.StructField(title_type, t.StringType(), True),
                                  t.StructField(title_primary, t.StringType(), True),
                                  t.StructField(title_original, t.StringType(), True),
                                  t.StructField(title_adult, t.IntegerType(), True),
                                  t.StructField(title_start_year, t.IntegerType(), True),
                                  t.StructField(title_end_year, t.IntegerType(), True),
                                  t.StructField(title_minutes, t.StringType(), True),
                                  t.StructField(title_genres, t.StringType(), True)])

    # Create basics dataframe from csv file, where separator is \t, columns has headers and null value is \N
    titles_df = spark_session.read.csv(titles_path, schema=titles_schema, sep='\t', header=True, nullValue="\\N")

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

    return names_df, titles_df, crew_df, ratings_df


names_df, titles_df, crew_df, ratings_df = create_dataframes()


def analyse_data():
    # Show first 20 rows of each dataframe
    names_df.show()
    titles_df.show()
    crew_df.show()
    ratings_df.show()

    # Show schema of each dataframe
    names_df.printSchema()
    titles_df.printSchema()
    crew_df.printSchema()
    ratings_df.printSchema()

    # Show rows number of each dataframe
    print('names:', names_df.count())
    print('titles:', titles_df.count())
    print('crew:', crew_df.count())
    print('ratings:', ratings_df.count())

    # Show statistics for columns of each dataframe
    names_df.describe().show()
    titles_df.describe().show()
    crew_df.describe().show()
    ratings_df.describe().show()

    # Show statistics about numerical columns
    names_df.select(name_birth_year, name_death_year).summary().show()
    titles_df.select(title_adult, title_start_year, title_end_year, title_minutes).summary().show()
    ratings_df.select(ratings_average, ratings_votes).summary().show()


analyse_data()


