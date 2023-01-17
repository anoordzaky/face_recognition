from decouple import config


from flask import Response
from pymongo import MongoClient


class Database:
    def __init__(self):

        # initialize connection to MongoDB
        print("Connecting to MongoDB Cluster..")

        self.URI = config("MONGO_URI")
        self.collection = config("COLLECTION")
        self.db_name = config("DB_NAME")

        self.db_cluster = MongoClient(self.URI)
        self.db = self.db_cluster[f'{self.db_name}'][f'{self.collection}']

    def push_files(self, package: dict):

        assert type(package) == dict, "Input must be a dictionary"
        try:
            self.db.insert_one(package)
        except:
            print("Upload Failed!")
            return Response("Upload Failed", 500)
        else:
            print("Upload Success!")

    def get_one_file(self, argument):

        # method to get one file from database, argument is not required
        if argument:
            assert type(
                argument) == dict, "Argument must be a dictionary (JSON)."
            request = self.db.find_one(argument)
        else:
            request = self.db.find_one()

        package = {}

        for i in request:
            package[i['_id']] = i

        return package

    def get_all_files(self):
        # method to get all files in the database
        package = {}
        request = self.db.find()

        for i in request:
            package[i['_id']] = i

        return package

    def get_image_names(self):
        # method to get the names and image to a dictionary with the shape of {"name": [], "embedding": []}

        package = {"name": [],
                   "embedding": []
                   }

        all = self.get_all_files()

        for key in all:
            package["name"].append(all[key]["name"])
            package["embedding"].append(all[key]["embedding"])

        return package
