import logging
import random
import string
from openai import OpenAI
from tenacity import retry, wait_random, stop_after_attempt


wait_random_min = 10
wait_random_max = 60
stop_after_attempt_num = 20

def generate_random_string(length=6):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

log_file = generate_random_string()
logger = logging.getLogger(log_file)
logger.setLevel(logging.INFO)
file_handler2 = logging.FileHandler(f"log/{log_file}.log")
formatter2 = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler2.setFormatter(formatter2)
logger.addHandler(file_handler2)
logger.propagate = False

def log_error(retry_state):
    attempt_number = retry_state.attempt_number
    exception = retry_state.outcome.exception()

    if exception:
        logger.info(f"Attempt {attempt_number} failed with exception: {exception}")

class OpenAIClient:
    def __init__(self, api_key=None, url=None) -> None:
        self.client = OpenAI(api_key=api_key)

    @retry(
        wait=wait_random(min=wait_random_min, max=wait_random_max),
        stop=stop_after_attempt(stop_after_attempt_num),
        after=log_error
    )
    def completion_with_backoff(self, **kwargs):
        messages = kwargs['messages']
        sys_content = messages[0]['content']
        user_content = messages[1]['content']
        messages = [
            {"role": "user", "content": f"{sys_content}\n\n{user_content}"}
        ]
        n = kwargs.get('n', 1)

        config ={
            "model": kwargs["model"],
            "messages": messages,
            "n": n
        }

        full_response = self.client.chat.completions.create(**config)

        responses = []
        try:
            for choice in full_response.choices:
                responses.append(choice.message.content)
        except Exception as e:
            logger.info(f"Attempt failed with exception: {e}: {str(full_response)}")
            raise e

        return responses
