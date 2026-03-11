import os

from openai import OpenAI
import pymongo
from pymongo import UpdateOne
from tqdm import tqdm
from pymongo.operations import SearchIndexModel


class RAGService:
    def __init__(
        self,
        embedding_model: str,
        query_model: str,
        db_url: str,
        collection: str,
    ):
        self.openai_client = OpenAI()
        self.embedding_model = embedding_model
        self.query_model = query_model
        self.db_url = db_url
        self.db_client = pymongo.MongoClient(self.db_url)
        self.collection = (
            self.db_client[os.getenv("MONGO_DB_NAME")][collection]
            if collection
            else self.db_client[os.getenv("MONGO_DB_NAME")][
                os.getenv("MONGO_COLLECTION_NAME_RECIPES")
            ]
        )

    def generate_response(self, anfrage: str):
        PROMPT = "Answer the given query using metric measurements, to the best of your abilities, using the input query and the information provided under <Recipies>. Tell the user which recipes were retrieved and argue why you like or dislike each of them, given their request. Finish with a recommendation that should best suit the user and the full recipe for that dish.\n\n<User Request>\n{anfrage}\n </User Request>\n\n<Recipes>"

        PROMPT += f"User Request: {anfrage}\n\n"
        retrieved_recipes = self.query_database(anfrage)

        for rec in retrieved_recipes:
            PROMPT += f"Title: {rec['title']}\nIngredients: {rec['ingredients']}\nInstructions:\n{rec['instructions']}\n\n"
        PROMPT += "</Recipes>"

        response = self.openai_client.chat.completions.create(
            model=self.query_model, messages=[{"role": "user", "content": PROMPT}]
        )
        return response.choices[0].message.content

    def query_database(self, anfrage):
        query_embedding = self.get_embedding(anfrage)
        pipeline = self.create_vector_search_pipeline(query_embedding)

        return self.collection.aggregate(pipeline)

    def setup(self):
        documents = self.retrieve_documents()

        self.embed_documents(documents)

        self.collection.create_search_index(self.create_search_index_model())

    def get_embedding(self, text):
        embedding = (
            self.openai_client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            .data[0]
            .embedding
        )
        return embedding

    def retrieve_documents(self):
        return self.collection.find({})

    def embed_documents(self, documents):
        updated_doc_count = 0
        # Generate the list of bulk write operations
        operations = []
        for doc in tqdm(documents):
            ingredients = "".join(doc["ingredients"])
            # Generate embeddings for this document
            embedding = self.get_embedding(ingredients)

            # Add the update operation to the list
            operations.append(
                UpdateOne({"_id": doc["_id"]}, {"$set": {"embedding": embedding}})
            )
        # Execute the bulk write operation
        if operations:
            result = self.collection.bulk_write(operations)
            updated_doc_count = result.modified_count
            print(f"Updated {updated_doc_count} documents.")

    def create_search_index_model(self):
        # hier setzen wir die relevanten Parameter für unser Retrieval
        return SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "similarity": "dotProduct",
                        "numDimensions": 1536,
                    }
                ]
            },
            name="vector_index",
            type="vectorSearch",
        )

    def create_vector_search_pipeline(self, query_embedding, anzahl_resultate=3):
        return [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "exact": True,
                    "limit": anzahl_resultate,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": 1,
                    "ingredients": 1,
                    "instructions": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
