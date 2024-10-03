import pandas as pd
import chromadb
import uuid
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Portfolio:
    def __init__(self, file_path=None):
        logger.info("Initializing Portfolio...")
        if file_path is None:
            file_path = r"C:\email_generator\.ipynb_checkpoints\app\resource\my_portfolio.csv"
        
        self.file_path = os.path.abspath(file_path)
        logger.info(f"Using file path: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The specified file does not exist: {self.file_path}")
        
        logger.info("Loading CSV file...")
        self.data = pd.read_csv(self.file_path)
        logger.info(f"CSV file loaded. Shape: {self.data.shape}")

        logger.info("Initializing ChromaDB...")
        chroma_db_path = os.path.join(os.path.dirname(self.file_path), "vectorstore")
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
        logger.info("ChromaDB initialized.")

    def load_portfolio(self):
        logger.info("Loading portfolio...")
        if not self.collection.count():
            logger.info("Collection is empty. Adding data...")
            for _, row in self.data.iterrows():
                if 'Techstack' in row and 'Links' in row:
                    self.collection.add(
                        documents=[row["Techstack"]],
                        metadatas={"links": row["Links"]},
                        ids=[str(uuid.uuid4())]
                    )
            logger.info("Portfolio data loaded successfully.")
        else:
            logger.info("Portfolio data already loaded.")
        logger.info(f"Collection count: {self.collection.count()}")

    def query_links(self, skills):
        logger.info(f"Querying links for skills: {skills}")
        if not skills:
            raise ValueError("You must provide skills for the query.")
        if isinstance(skills, str):
            skills = [skills]  # Convert single skill to list
        results = self.collection.query(query_texts=skills, n_results=2)
        logger.info(f"Query results: {results}")
        return results.get('metadatas', [])

    def get_all_skills(self):
        logger.info("Getting all skills...")
        skills = self.data['Techstack'].tolist() if 'Techstack' in self.data.columns else []
        logger.info(f"Found {len(skills)} skills.")
        return skills