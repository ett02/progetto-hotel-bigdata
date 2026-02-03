import time
import os
import sys

# Add current directory to path so we can import backend
sys.path.append(os.path.dirname(__file__))

print("Importing backend...")
try:
    from backend import GestoreBigData
except ImportError as e:
    print(f"ERROR importing backend: {e}")
    sys.exit(1)

print("Initializing GestoreBigData...")
start_time = time.time()
gestore = GestoreBigData()

print("Getting Spark Session...")
try:
    spark = gestore.get_spark_session()
    print(f"Spark Session initialized in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"ERROR initializing Spark: {e}")
    sys.exit(1)

PATH_HOTEL = r"f:\prog_big_1\_dati\Hotel_Reviews.csv"
PATH_IMDB = r"f:\prog_big_1\_dati\IMDB Dataset.csv"

print(f"Loading Hotel Data from {PATH_HOTEL}...")
load_start = time.time()
try:
    df_hotel = gestore.carica_dati_hotel(PATH_HOTEL)
    count = df_hotel.count()
    print(f"Hotel Data Loaded. Row count: {count}")
    print(f"Time taken: {time.time() - load_start:.2f} seconds.")
except Exception as e:
    print(f"ERROR loading Hotel Data: {e}")

print(f"Loading IMDB Data from {PATH_IMDB}...")
load_start = time.time()
try:
    df_imdb = gestore.carica_dati_imdb(PATH_IMDB)
    count = df_imdb.count()
    print(f"IMDB Data Loaded. Row count: {count}")
    print(f"Time taken: {time.time() - load_start:.2f} seconds.")
except Exception as e:
    print(f"ERROR loading IMDB Data: {e}")

print("Test Completed.")
