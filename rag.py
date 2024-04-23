from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from utils import get_random_name
from bs4 import BeautifulSoup
from config import Config
import numpy as np
import faiss


# implements RAG indexer
class Indexer:
	def __init__(self, config: Config):
		self.config = config
		self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
		emb = self.model.encode('hello world', precision=self.config.PRECISION)

		self.index = faiss.IndexFlatL2(emb.shape[0])
		self.store = []
	
	# add paragraphs from a text (text separated with \n), a document (path to file locally), or url (uri to website)
	def add(self, content:str = None, label:str = None, doc:str = None, url:str = None):

		if label == None:
			label = 'undefined'

		if doc != None:
			with open(doc, 'r') as file:
				for line in file:
					if len(line) > self.config.MIN_PARAGRAPH_LENGTH:
						self.index.add(self.model.encode([line.strip()], precision=self.config.PRECISION))
						self.store.append((label, line))

		if url != None:
			filename = self.config.TMP_PATH + get_random_name()

			util.http_get(url, filename)

			with open(filename) as file:
				soup = BeautifulSoup(file.read(), 'html.parser')

				for line in soup.get_text().split('\n'):
					if len(line) > self.config.MIN_PARAGRAPH_LENGTH:
						self.index.add(self.model.encode([line.strip()], precision=self.config.PRECISION))
						self.store.append((label, line))

		if content != None:
			for line in content.split('\n'):
				if len(line) > self.config.MIN_PARAGRAPH_LENGTH:
					self.index.add(self.model.encode([line.strip()], precision=self.config.PRECISION))
					self.store.append((label, line))
				
	# retrieve information about a specific paragraph
	def retrieve(self, id: int) -> tuple:
		return self.store[id]

	# search for most similar paragraphs
	def search(self, query: str, label:str = None, top:int = 5) -> list:
		_, ids = self.index.search(np.array([self.model.encode(query)]), top)

		return ids[0].tolist()
