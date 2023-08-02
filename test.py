import unittest

from schemas import *
from pyspark_test import assert_pyspark_df_equal
from pyspark.sql import SparkSession
from main import (
    create_dataframes,
    analyse_data,
    directors_rating,
    genres_rating,
    top_3_by_type,
    top_20_by_length,
    length_rating,
    rating_statistics,
    write_csv,
)


class TestMyModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("Test") \
            .master("local[1]") \
            .getOrCreate()
        cls.test_names_path = './tests/test_name.basics.tsv.gz'
        cls.test_titles_path = './tests/test_title.basics.tsv.gz'
        cls.test_crew_path = './tests/test_title.crew.tsv.gz'
        cls.test_ratings_path = './tests/test_title.ratings.tsv.gz'

        # Prepare test data for names_df
        columns_names = [name_id, name_primary_name, name_birth_year, name_death_year, name_primary_profession,
                         name_known_for]

        data_names = [
            ("nm0000001", "Fred Astaire", 1899, 1987, "soundtrack,actor,miscellaneous",
             "tt0072308,tt0031983,tt0050419,tt0053137"),
            ("nm0000002", "Lauren Bacall", 1924, 2014, "actress,soundtrack",
             "tt0117057,tt0075213,tt0038355,tt0037382"),
            ("nm0000003", "Brigitte Bardot", 1934, None, "actress,soundtrack,music_department",
             "tt0056404,tt0054452,tt0049189,tt0057345"),
            ("nm0000004", "John Belushi", 1949, 1982, "actor,soundtrack,writer",
             "tt0077975,tt0078723,tt0072562,tt0080455"),
            ("nm0000005", "Ingmar Bergman", 1918, 2007, "writer,director,actor",
             "tt0083922,tt0050976,tt0069467,tt0050986")
        ]

        # Prepare test data for titles_df
        columns_titles = [title_id, title_type, title_primary, title_original, title_adult, title_start_year,
                          title_end_year, title_minutes, title_genres]

        data_titles = [
            ("tt0000001", "short", "Carmencita", "Carmencita", 0, 2021, None, "1", "Documentary,Short"),
            ("tt0000002", "short", "Le clown et ses chiens", "Le clown et ses chiens", 0, 2021, None, "5",
             "Animation,Short"),
            ("tt0000003", "short", "Pauvre Pierrot", "Pauvre Pierrot", 0, 2021, None, "4",
             "Animation,Comedy,Romance"),
            ("tt0000004", "short", "Un bon bock", "Un bon bock", 0, 2021, None, "12", "Animation,Short"),
            ("tt0000005", "short", "Blacksmith Scene", "Blacksmith Scene", 0, 2021, None, "1", "Comedy,Short"),
            ("tt0000006", "short", "Chinese Opium Den", "Chinese Opium Den", 0, 2021, None, "1", "Short"),
            ("tt0000007", "short", "Corbett and Courtney Before the Kinetograph",
             "Corbett and Courtney Before the Kinetograph", 0, 2031, None, "1", "Short,Sport"),
            ("tt0000008", "short", "Edison Kinetoscopic Record of a Sneeze", "Edison Kinetoscopic Record of a Sneeze",
             0, 2031, None, "1", "Documentary,Short"),
            ("tt0000009", "movie", "Miss Jerry", "Miss Jerry", 0, 2031, None, "45", "Romance"),
            ("tt0000010", "short", "Leaving the Factory", "La sortie de l'usine Lumière à Lyon", 0, 2031, None,
             "1", "Documentary,Short"),
            ("tt0000011", "short", "Akrobatisches Potpourri", "Akrobatisches Potpourri", 0, 2031, None, "1",
             "Documentary,Short"),
            ("tt0000012", "short", "The Arrival of a Train", "L'arrivée d'un train à La Ciotat", 0, 2025, None,
             "1", "Documentary,Short"),
            ("tt0000013", "short", "The Photographical Congress Arrives in Lyon",
             "Le débarquement du congrès de photographie à Lyon", 0, 2025, None, "1", "Documentary,Short"),
            ("tt0000014", "short", "The Waterer Watered", "L'arroseur arrosé", 0, 2025, None, "1", "Comedy,Short"),
            ("tt0000015", "short", "Autour d'une cabine", "Autour d'une cabine", 0, 2025, None, "2",
             "Animation,Short")
        ]

        # Prepare test data for crew_df
        columns_crew = [title_id, crew_directors, crew_writers]

        data_crew = [
            ("tt0000001", "nm0000001", None),
            ("tt0000002", "nm0000001", None),
            ("tt0000003", "nm0000001", None),
            ("tt0000004", "nm0000002", None),
            ("tt0000005", "nm0000005", None),
            ("tt0000006", "nm0000004", None),
            ("tt0000007", "nm0000004,nm0000001", None),
            ("tt0000008", "nm0000002", None),
            ("tt0000009", None, "nm0000005"),
            ("tt0000010", "nm0000000", None),
            ("tt0000011", "nm0000004", None),
            ("tt0000012", "nm0000000,nm0000008", None),
            ("tt0000013", "nm0000004", None),
            ("tt0000014", "nm0000004", None),
            ("tt0000015", "nm0000004", None)
        ]

        # Prepare test data for ratings_df
        columns_ratings = [title_id, ratings_average, ratings_votes]

        data_ratings = [
            ("tt0000001", 9.2, 1989),
            ("tt0000002", 5.8, 265),
            ("tt0000003", 9.2, 1850),
            ("tt0000004", 5.5, 178),
            ("tt0000005", 9.2, 2635),
            ("tt0000006", 5.1, 182),
            ("tt0000007", 9.2, 825),
            ("tt0000008", 8.4, 2128),
            ("tt0000009", 8.3, 206),
            ("tt0000010", 8.9, 7229),
            ("tt0000011", 8.3, 369),
            ("tt0000012", 7.4, 12360),
            ("tt0000013", 5.7, 1897),
            ("tt0000014", 8.1, 5561),
            ("tt0000015", 6.2, 1097)
        ]

        # Prepare and write test dataframes
        cls.test_names_df = cls.spark.createDataFrame(data_names, schema=names_schema).toDF(*columns_names)
        cls.test_titles_df = cls.spark.createDataFrame(data_titles, schema=titles_schema).toDF(*columns_titles)
        cls.test_crew_df = cls.spark.createDataFrame(data_crew, schema=crew_schema).toDF(*columns_crew)
        cls.test_ratings_df = cls.spark.createDataFrame(data_ratings, schema=ratings_schema).toDF(*columns_ratings)
        cls.test_names_df.write.csv(cls.test_names_path, header=True, mode='overwrite')
        cls.test_titles_df.write.csv(cls.test_titles_path, header=True, mode='overwrite')
        cls.test_crew_df.write.csv(cls.test_crew_path, header=True, mode='overwrite')
        cls.test_ratings_df.write.csv(cls.test_ratings_path, header=True, mode='overwrite')

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_create_dataframes(self):
        names_df, titles_df, crew_df, ratings_df = create_dataframes(
            self.test_names_path, self.test_titles_path, self.test_crew_path, self.test_ratings_path
        )
        self.assertNotEqual(names_df.count(), 0)
        self.assertNotEqual(titles_df.count(), 0)
        self.assertNotEqual(crew_df.count(), 0)
        self.assertNotEqual(ratings_df.count(), 0)
        self.assertEqual(names_df.count(), self.test_names_df.count())
        self.assertEqual(titles_df.count(), self.test_titles_df.count())
        self.assertEqual(crew_df.count(), self.test_crew_df.count())
        self.assertEqual(ratings_df.count(), self.test_ratings_df.count())

    def test_analyse_data(self):
        # As the `analyse_data` function only displays information and doesn't return anything,
        # we can't directly test the output. Instead, we'll just check that it runs without any errors.
        try:
            analyse_data(self.test_names_df, self.test_titles_df, self.test_crew_df, self.test_ratings_df)
        except Exception as e:
            self.fail(f"Failed to run `analyse_data`: {e}")

    def test_directors_rating(self):
        # Call the `directors_rating` function with the test dataframes
        result_df = directors_rating(self.test_crew_df, self.test_ratings_df, self.test_names_df)
        result_df.show()
        self.assertEqual(result_df.count(), 4)
        self.assertEqual(result_df.head()[name_primary_name], 'Fred Astaire')

    def test_genres_rating(self):
        # Call the `genres_rating` function with the test dataframe
        result_df = genres_rating(self.test_titles_df)

        self.assertEqual(result_df.count(), 6)
        self.assertEqual(result_df.head()[genre], 'Short')

    def test_top_3_by_type(self):
        # Call the `top_3_by_type` function with the test dataframes
        result_df = top_3_by_type(self.test_titles_df, self.test_ratings_df)

        self.assertEqual(result_df.count(), 4)
        self.assertEqual(result_df.head()[title_primary], 'Miss Jerry')

    def test_top_20_by_length(self):
        # Call the `top_20_by_length` function with the test dataframe
        result_df = top_20_by_length(self.test_titles_df, self.test_ratings_df)

        self.assertEqual(result_df.count(), 4)
        self.assertEqual(result_df.head()[title_minutes], '4')

    def test_length_rating(self):
        # Call the `length_rating` function with the test dataframes
        result_df = length_rating(self.test_titles_df, self.test_ratings_df)

        self.assertEqual(result_df.count(), 10)
        self.assertEqual(result_df.head()[title_primary], 'Pauvre Pierrot')

    def test_rating_statistics(self):
        # Call the `rating_statistics` function with the test dataframe
        result_df = rating_statistics(self.test_ratings_df)

        self.assertEqual(result_df.count(), 11)
        self.assertEqual(round(result_df.head()[ratings_average], 2), 9.2)

    def test_write_csv(self):
        # Call the `write_csv` function with the test dataframe
        write_csv(self.test_ratings_df, 'test_ratings_df')

        test_path = './results/test_ratings_df'

        result_df = self.spark.read.csv(test_path, schema=ratings_schema, header=True)

        assert_pyspark_df_equal(result_df, self.test_ratings_df)


if __name__ == "__main__":
    unittest.main()

