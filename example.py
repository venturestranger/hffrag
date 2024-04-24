from hffrag import Templater, Driver, Indexer
from config import DriverConfig


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
	('system', 'You are a told to say a few words about your job relying on the following information: {information}'),
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
