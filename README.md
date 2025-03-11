# HFFRAG - RAG using HF (sentence transformers) and Faiss (similarity search library)

The project implements RAG (Retrieval-Augmented Generation) and introduces several components for indexing documents, querying, finding the most similar documents, handling templates, and generating outputs. The project supports both `Ollama` and `OpenAI` LLM agents.

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

The `hffrag` implements several components:

## 1. Indexer

The `Indexer` class implements an indexer for RAG approach. It allows you to add paragraphs from various sources, such as text, documents, or URLs, and provides methods for retrieving information about specific paragraphs and searching for the most similar ones.

### Usage

Here's how you can use the `Indexer` class:

```python
from hffrag import Indexer
from hffrag.config import IndexerConfig

# Create a configuration object
config = IndexerConfig()

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

The `Driver` implements an interface to initialize connection with an LLM (powered with `Ollama`) and query the model in a number of ways:

```python
from hffrag import Templater, Driver
from hffrag.config import DriverConfig

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

### Asynchronous querying

You can also use the command below, but only if there is `async_requests` specified.
`async_requests` should implement an http client, attributing async `.post(url, json=data)` method
that returns an arbitrary response instance. The response object (from which the response instance was originated) should contain async `.json()` method that yields a parsed body content:

```python
output = await llm.aquery('What do you do', template=temp, async_requests=async_requests)
```

### Querying with enabled streaming

You can also use `.squery(...)` method for prompting an LLM. In this case, it will stream generated tokens in real-time:

```python
for output in llm.squery('What do you do', template=temp):
	print(output)
```

### Using a custom LLM endpoint 

In case if you wish to request an arbitrary `Ollama` endpoint, assign a new url to your `url_token` field:

```python
output = llm.query('What do you do', url_token='http://localhost:11434/api/generate')
```

### Using OpenAI agents 

To access `OpenAI` API and base your driver on some of their models, you should set `llm_type` to `openai` and `url_token` to your OpenAI API token:

```python
output = llm.query('What do you do', url_token='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', llm_type='openai')
```

OpenAI agents also work with for asynchronous queries and queries with enabled streaming.

## Example of Usage

```python
from hffrag import Templater, Driver, Indexer
from hffrag.config import DriverConfig


# initialize retriever
doc = """
IT engineers are the backbone of the technological world, ensuring the smooth operation and functionality of computer systems, networks, and applications. Their responsibilities encompass a wide range of tasks, from designing and implementing new systems to troubleshooting technical problems and keeping everything up-to-date.

Writers are the architects of the written word. They wield language as their tool, crafting stories, poems, articles, scripts, and countless other forms of creative expression. Their work entertains, informs, educates, and inspires, shaping our understanding of the world and ourselves.
"""

index = Indexer()
index.add(doc, label='career_path')


# initialize a template
template = Templater([
	('system', 'You are a {specialization}'),
	('system', 'You are told to say a few words about your job relying on the following information: {information}'),
	('human', 'What do you do?')
])


# retrieve information about the specified specialization
specialization = 'IT engineer'
information = index.retrieve(index.search('IT engineer', top=1)[0])[1]


# intialize LLM
driver_config = DriverConfig()
driver_config.LLM_MODEL = 'mistral:7b-instruct'

llm = Driver(driver_config)


# query LLM
print(llm.query(template=template, specialization=specialization, information=information))
```

### Output
The output is a json document, containing fields `response` (string) and `done` (boolean). The `response` field carries generated tokens. The `done` field returns whether generation is finished.

```json
{"response": "As an IT engineer, I play a crucial role in the technological world by maintaining and enhancing the functionality of computer systems, networks, and applications. My job involves a diverse range of tasks, from designing and implementing new systems to troubleshooting technical problems and keeping up-to-date with the latest technologies. I work behind the scenes to ensure that the technology infrastructure is running smoothly and efficiently, enabling businesses and organizations to operate effectively. My goal is to prevent technical issues before they occur and to quickly resolve any problems that do arise, minimizing downtime and maximizing productivity.", "done": true}
```
