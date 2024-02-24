# imports
import time  # for measuring time duration of API calls
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))


def noStreamResponse():
    # record the time before the request is sent
    start_time = time.time()

    # send a ChatCompletion request to count to 100
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user',
             'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
        ],
        temperature=0,
    )
    # calculate the time it took to receive the response
    response_time = time.time() - start_time

    # print the time delay and text received
    print(f"Full response received {response_time:.2f} seconds after request")
    print(f"Full response received:\n{response}")

def streamResponse():
    # record the time before the request is sent
    start_time = time.time()

    # send a ChatCompletion request to count to 100
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user',
             'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
        ],
        temperature=0,
        stream=True  # again, we set stream=True
    )
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

    # print the time delay and text received
    print(f"Full response received {chunk_time:.2f} seconds after request")
    # clean None in collected_messages
    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join([m for m in collected_messages])
    print(f"Full conversation received: {full_reply_content}")

def run():
    noStreamResponse()
    streamResponse()



