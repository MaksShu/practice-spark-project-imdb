import pyspark
from columns import *

from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
import pyspark.sql.types as t
import pyspark.sql.functions as f

# Initialize Spark Session
spark_session = (SparkSession.builder
                             .master("local")
                             .appName("task app")
                             .config(conf=SparkConf())
                             .getOrCreate())

spark_session.catalog.clearCache()

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


# Calculate rating of directors by amount of titles with rating more than 8
def directors_rating():
    # Prepare data
    # Remove rows where directors column value is null and parse directors column into separate rows
    directors_df = crew_df.filter(f.col(crew_directors).isNotNull())\
                .select(title_id, f.explode(f.split(f.col(crew_directors), ",")).alias(director))
    # Get titles with rating more than 8
    min_rating = 8
    ratings_more_than_8_df = ratings_df.filter(f.col(ratings_average) > min_rating)

    # Join ratings_more_than_8_df and directors_df by title id, taking only director id and title id columns,
    # then group by director and count amount of titles and average rating
    directors_rating_df = directors_df.join(ratings_more_than_8_df,
                                            ratings_more_than_8_df[title_id] == directors_df[title_id])\
                                      .select(director, directors_df[title_id], ratings_average)\
                                      .groupBy(director)\
                                      .agg(f.count(f.col(title_id)).alias(titles_count),
                                           f.round(f.avg(f.col(ratings_average)), 2).alias(ratings_average))\
                                      .cache()

    # Join directors_rating_df and names_df(only name id, primary name and death year columns) by name id and director
    # then rearrange columns into order director id, primary name, titles count and new column isAlive,
    # where 1 means person is alive, so death year is null, and 0 means opposite,
    # then order by titlesCount and averageRating descending
    directors_rating_df = directors_rating_df.join(names_df.select(name_id, name_primary_name, name_death_year),
                                                   f.col(name_id) == f.col(director))\
                                             .select(director, name_primary_name, titles_count, ratings_average,
                                                     f.when(f.col(name_death_year).isNotNull(), 0).otherwise(1)
                                                     .alias(alive))\
                                             .orderBy(f.col(titles_count), f.col(ratings_average), ascending=False)

    directors_rating_df.cache().show(truncate=False)
    print('All directors: ', directors_rating_df.count())
    print('Alive directors: ', directors_rating_df.filter(f.col(alive) == 1).count())
    return directors_rating_df


directors_rating_df = directors_rating()


# Calculate statistics over amount of films that started after 2020 by genres
def genres_rating():
    # Prepare data
    # Remove rows where start year or genres column value is null,
    # then filter titles by start year,
    # then parse genres column into separate rows,
    # then group by genre and count titlesCount,
    # then order by titlesCount
    min_year = 2020
    titles_by_genres_df = titles_df.filter(f.col(title_start_year).isNotNull() & f.col(title_genres).isNotNull())\
                                   .filter(f.col(title_start_year) > min_year)\
                                   .select(title_id, f.explode(f.split(f.col(title_genres), ",")).alias(genre))\
                                   .groupBy(genre)\
                                   .agg(f.count(f.col(title_id)).alias(titles_count))\
                                   .orderBy(f.col(titles_count), ascending=False)

    print(titles_by_genres_df.cache().count())
    titles_by_genres_df.show(50, truncate=False)

    return titles_by_genres_df


titles_by_genres_df = genres_rating()


# Find top 3 titles of each type by rating and votes number
def top_3_by_type():
    # Create window to go through titles by type
    # and sort them by rating and number of votes descending in each partition
    window = Window.partitionBy(title_type).orderBy(f.col(ratings_average).desc(), f.col(ratings_votes).desc())

    # Remove rows where type is null
    # then join titles dataframe with ratings
    # then create new column with ranking for each type using window
    # then leave only top 3 titles for each type
    top = 3
    top_3_by_type_df = titles_df.filter(f.col(title_type).isNotNull())\
                                         .join(ratings_df, titles_df[title_id] == ratings_df[title_id])\
                                         .withColumn(rank, f.rank().over(window))\
                                         .filter(f.col(rank) <= top)\
                                         .select(titles_df[title_id], title_type, title_primary,  ratings_average,
                                                 ratings_votes, rank)

    top_3_by_type_df.show(100, truncate=False)

    return top_3_by_type_df


top_3_by_type_df = top_3_by_type()


# Among titles with rating more than 9 show first 20 by length
def top_20_by_length():
    # Remove rows where runtimeMinutes is null
    # then join titles and rating dataframes on title id
    # then order by runtimeMinutes in descending order
    # then limit result to 20 rows and cache to save time with next operations
    min_rating = 9
    top_20_by_length_df = titles_df.filter(f.col(title_minutes).isNotNull())\
                                   .join(ratings_df.filter(f.col(ratings_average) > min_rating),
                                         ratings_df[title_id] == titles_df[title_id])\
                                   .orderBy(f.col(title_minutes).cast('int').desc())\
                                   .select(titles_df[title_id], title_primary, title_minutes, ratings_average)\
                                   .limit(20)\
                                   .cache()

    top_20_by_length_df.show(truncate=False)

    return top_20_by_length_df


top_20_by_length_df = top_20_by_length()


# Among titles with equal length find average rating and show top 5 by rating for each length
def length_rating():
    # Create window to go through titles by length
    # and sort them by rating and vote number descending in each partition
    window = Window.partitionBy(title_minutes).orderBy(f.col(ratings_average).desc(), f.col(ratings_votes).desc())
    window_spec = Window.partitionBy(title_minutes)

    # Remove rows where runtimeMinutes is null
    # then join titles dataframe with ratings on title id
    # then create new column with average rating for each length using window
    # then create new column with ranking for each length using window
    # then leave only top 3 titles for each type
    top = 5
    length_rating_df = titles_df.filter(f.col(title_minutes).isNotNull())\
                                .join(ratings_df, titles_df[title_id] == ratings_df[title_id])\
                                .withColumn(average_rating_length,
                                            f.round(f.avg(f.col(ratings_average)).over(window_spec), 2))\
                                .withColumn(rank, f.rank().over(window))\
                                .filter(f.col(rank) <= top)\
                                .select(titles_df[title_id], title_minutes, title_primary, ratings_average,
                                        average_rating_length, rank)\
                                .orderBy(f.col(average_rating_length).desc())\
                                .cache()

    length_rating_df.show(100)
    print(length_rating_df.count())

    return length_rating_df


length_rating_df = length_rating()


# For each rating calculate overall amount of titles and votes
def rating_statistics():
    # Group dataframe by ratings and count titles and sum of votes for each rating
    # then order by rating in descending order
    rating_stats_df = ratings_df.groupBy(ratings_average)\
                                .agg(f.count(title_id).alias(titles_count), f.sum(ratings_votes).alias(ratings_votes))\
                                .orderBy(f.col(ratings_average).desc())\
                                .cache()

    rating_stats_df.show(100)

    return rating_stats_df


rating_stats_df = rating_statistics()


def write_csv(df: pyspark.sql.DataFrame, name):
    path = './results/' + name
    df.write.csv(path, header=True, mode='overwrite')


write_csv(directors_rating_df, 'directors_rating_df')
write_csv(titles_by_genres_df, 'titles_by_genres_df ')
write_csv(top_3_by_type_df, 'top_3_by_type_df')
write_csv(top_20_by_length_df, 'top_20_by_length_df')
write_csv(length_rating_df, 'length_rating_df')
write_csv(rating_stats_df, 'rating_stats_df')
