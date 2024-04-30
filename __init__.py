from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from .utils import get_random_name
from bs4 import BeautifulSoup
from .config import IndexerConfig, DriverConfig
import numpy as np
import requests
import faiss


# implements RAG indexer
class Indexer:
	def __init__(self, config: IndexerConfig = None, embedding: SentenceTransformer = None):
		if config == None:
			self.config = IndexerConfig()
		else:
			self.config = config

		if embedding != None:
			self.model = embedding
		else:
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
				data = list(filter(lambda x: len(x) > self.config.MIN_PARAGRAPH_LENGTH, file.read()[:self.config.DOC_MAX_LENGTH].split('\n')))
				self.store.extend(zip([label for i in range(len(data))], data))

				self.index.add(self.model.encode(data, precision=self.config.PRECISION))

		if url != None:
			filename = self.config.TMP_PATH + get_random_name()

			util.http_get(url, filename)

			with open(filename) as file:
				soup = BeautifulSoup(file.read(), 'html.parser')

				data = list(filter(lambda x: len(x) > self.config.MIN_PARAGRAPH_LENGTH, soup.get_text()[:self.config.DOC_MAX_LENGTH].split('\n')))
				self.store.extend(zip([label for i in range(len(data))], data))

				self.index.add(self.model.encode(data, precision=self.config.PRECISION))

		if content != None:
			data = list(filter(lambda x: len(x) > self.config.MIN_PARAGRAPH_LENGTH, content[:self.config.DOC_MAX_LENGTH].split('\n')))
			self.store.extend(zip([label for i in range(len(data))], data))

			self.index.add(self.model.encode(data, precision=self.config.PRECISION))
				
	# retrieve information about a specific paragraph
	def retrieve(self, id: int) -> tuple:
		return self.store[id]

	# search for most similar paragraphs
	def search(self, query: str, label:str = None, top:int = 5) -> list:
		_, ids = self.index.search(np.array([self.model.encode(query)]), top)

		return ids[0].tolist()


# implements a template used for quering llm
class Templater:
	def __init__(self, msgs: list):
		self.system = ''
		self.prompt = ''

		for msg in msgs:
			if msg[0] == 'system':
				self.system += msg[1] + '\n'
			else:
				self.prompt += msg[1] + '\n'


class Driver:
	def __init__(self, config: DriverConfig = None):
		if config == None:
			self.config = DriverConfig()
		else:
			self.config = config
	
	# if __prompt is specified, it queries llm with just __prompt, ignoring the template
	# otherwise it uses the specified template and substitutes template arguments with **kargs
	def query(self, __prompt: str = None, template: Templater = None, base_url: str = None, **kargs) -> str:
		system = None
		prompt = None

		if template != None and __prompt == None:
			system = template.system.format(**kargs)

		if template != None and __prompt == None:
			prompt = template.prompt.format(**kargs)
		else:
			prompt = __prompt

		params = {
			'model': self.config.LLM_MODEL,
			'prompt': prompt,
			'stream': False
		}

		if system != None:
			params.update({'system': system})

		if base_url != None:
			resp = requests.post(base_url, json=params)
		else:
			resp = requests.post(self.config.LLM_BASE_URL, json=params)

		content = resp.json()['response']
		return content
	
	# asynchronous query requests
	async def aquery(self, __prompt: str = None, template: Templater = None, base_url: str = None, async_requests = None, **kargs) -> str:
		system = None
		prompt = None

		if template != None and __prompt == None:
			system = template.system.format(**kargs)

		if template != None and __prompt == None:
			prompt = template.prompt.format(**kargs)
		else:
			prompt = __prompt

		params = {
			'model': self.config.LLM_MODEL,
			'prompt': prompt,
			'stream': False
		}

		if system != None:
			params.update({'system': system})

		if base_url != None:
			resp = await async_requests.post(base_url, json=params)
		else:
			resp = await async_requests.post(self.config.LLM_BASE_URL, json=params)

		content = (await resp.json())['response']
		return content
