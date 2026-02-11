from pymongo import MongoClient

from app.core.config import db_settings


class Database:
    def __init__(self):
        uri = db_settings.uri.encoded_string()
        db_client = MongoClient(uri)
        db_name = db_settings.name
        db = db_client[db_name]

        self.cards_collection = db[db_settings.cards_collection]
        self.embeddings_collection = db[db_settings.card_embeddings_collection]


database = Database()
