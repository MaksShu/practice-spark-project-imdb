from columns import *
import pyspark.sql.types as t

# Create schema for names dataset
names_schema = t.StructType([t.StructField(name_id, t.StringType(), False),
                             t.StructField(name_primary_name, t.StringType(), False),
                             t.StructField(name_birth_year, t.IntegerType(), True),
                             t.StructField(name_death_year, t.IntegerType(), True),
                             t.StructField(name_primary_profession, t.StringType(), True),
                             t.StructField(name_known_for, t.StringType(), True)])

# Create schema for titles dataset
titles_schema = t.StructType([t.StructField(title_id, t.StringType(), False),
                              t.StructField(title_type, t.StringType(), True),
                              t.StructField(title_primary, t.StringType(), True),
                              t.StructField(title_original, t.StringType(), True),
                              t.StructField(title_adult, t.IntegerType(), True),
                              t.StructField(title_start_year, t.IntegerType(), True),
                              t.StructField(title_end_year, t.IntegerType(), True),
                              t.StructField(title_minutes, t.StringType(), True),
                              t.StructField(title_genres, t.StringType(), True)])

# Create schema for crew dataset
crew_schema = t.StructType([t.StructField(title_id, t.StringType(), False),
                            t.StructField(crew_directors, t.StringType(), True),
                            t.StructField(crew_writers, t.StringType(), True)])

# Create schema for ratings dataset
ratings_schema = t.StructType([t.StructField(title_id, t.StringType(), False),
                               t.StructField(ratings_average, t.FloatType(), True),
                               t.StructField(ratings_votes, t.IntegerType(), True)])