class IndexerConfig:
	TMP_PATH = './tmp/'
	PRECISION = 'float32'
	MIN_PARAGRAPH_LENGTH = 10
	EMBEDDING_MODEL = 'distiluse-base-multilingual-cased-v1'
	DOC_MAX_LENGTH = 10000


class DriverConfig:
	LLM_BASE_URL = 'http://localhost:11434/api/generate'
	LLM_MODEL = 'mistral:7b-instruct'
