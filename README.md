# RAG using HF (sentence transformers) and Faiss (similarity search library)

The project implements RAG (Retrieval-Augmented Generation) and introduces several components for indexing documents, querying, finding the most similar ones, generating llm templates and outputs (using `ollama` software).

### Installation

To use the project, you need to install the required libraries:

```bash
pip install sentence_transformers faiss bs4
```

Or consider running the following program:

```bash
pip install -r ./requirements.txt
```

---

The `rag.py` file implements several components:

## 1. Indexer

The `Indexer` class implements an indexer for RAG approach. It allows you to add paragraphs from various sources, such as text, documents, or URLs, and provides methods for retrieving information about specific paragraphs and searching for the most similar ones.

### Usage

Here's how you can use the `Indexer` class:

```python
from config import Config
from indexer import Indexer

# Create a configuration object
config = Config()

# Initialize the Indexer
indexer = Indexer(config)

# Add paragraphs from different sources
indexer.add(content="Paragraph 1\nParagraph 2\nParagraph 3", label="Document 1")
indexer.add(doc="path/to/document.txt", label="Document 2")
indexer.add(url="https://example.com", label="Web Document")

# Search for the most similar paragraphs, returns ids
query = "Query paragraph"
similar_paragraphs = indexer.search(query, top=2)
print(similar_paragraphs)
# >> [0, 1]

# Retrieve information about a specific paragraph, returns a tuple(label, paragraph)
paragraph_id = 0
info = indexer.retrieve(paragraph_id)
print(info)
# >> ('Document 1', 'Paragraph 1')
```

## 2. Templater

The `Templater` class implements templates for querying LLMs. It provides an interface to wrap around specified prompt details as well as to build system and human messages to query LLMs.

## 3. Driver

The `Driver` implements an interface to initialize connection with an LLM (hosted with `Ollama`) and query the model in a number of ways:

```python
from rag import Templater, Driver
from config import DriverConfig

temp = Templater([
	('system', 'You are a {specialization}'),
	('system', 'You are a told to say a few words about your job'),
	('human', 'What do you do?')
])

llm = Driver(DriverConfig())

# querying LLM using a templater
print(llm.query(template=temp, specialization='IT engineer'))

# querying LLM using just the prompt
print(llm.query('What do you do'))

# querying LLM using just the prompt (it will ignore the template)
print(llm.query('What do you do', template=temp))
```  
