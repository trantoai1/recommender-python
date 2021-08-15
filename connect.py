from psycopg2 import connect
from dotenv import load_dotenv
import os

load_dotenv()
URL = os.getenv('url')
database = os.getenv('database')
username = os.getenv('username')
password = os.getenv('')
conn = connect(
    host=URL,
    database=database,
    user=username,
    password=password)
# cur = conn.cursor()
#
# # execute a statement
# print('PostgreSQL database version:')
# cur.execute('SELECT * from products')
# db_version = cur.fetchone()
# print(db_version)