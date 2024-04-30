import random


def get_random_name(format: str = 'txt', name_length: int = 30) -> str:
	alphabet = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890'

	name = ''.join([random.choice(alphabet) for i in range(name_length)]) + f'.{format}'

	return name
