from rag import Indexer
from config import Config


sentences = [
	'hello world', 
	'привет мир', 
	'hallo wereld',
	'salut le monde'
]


rag = Indexer(Config())

for i in sentences:
	rag.add(i)

rag.add(label='google', url='https://www.google.com')

ids = rag.search('здравсвуй ми', top=4)
print(ids)
print(rag.retrieve(ids[0]))
