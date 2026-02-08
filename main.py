import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGODB_URI")
print(f"Connecting to database: {uri}")
client = MongoClient(uri)

if __name__ == "__main__":
    print("Hello!")
