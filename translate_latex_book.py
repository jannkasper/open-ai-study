from openai import OpenAI
import os
from transformers import GPT2Tokenizer

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if you didn't set as an env var>"))


def group_chunks(chunks, ntokens, max_len=1000, hard_max_len=3000):
    """
    Group very short chunks, to form approximately page long chunks.
    """
    batches = []
    cur_batch = ""
    cur_tokens = 0

    # iterate over chunks, and group the short ones together
    for chunk, ntoken in zip(chunks, ntokens):
        # discard chunks that exceed hard max length
        if ntoken > hard_max_len:
            print(
                f"Warning: Chunk discarded for being too long ({ntoken} tokens > {hard_max_len} token limit). Preview: '{chunk[:50]}...'")
            continue

        # if room in current batch, add new chunk
        if cur_tokens + 1 + ntoken <= max_len:
            cur_batch += "\n\n" + chunk
            cur_tokens += 1 + ntoken  # adds 1 token for the two newlines
        # otherwise, record the batch and start a new one
        else:
            batches.append(cur_batch)
            cur_batch = chunk
            cur_tokens = ntoken

    if cur_batch:  # add the last batch if it's not empty
        batches.append(cur_batch)

    return batches


def translate_chunk(chunk, model='gpt-3.5-turbo',
                    dest_language='English',
                    sample_translation=("\poglavje{Osnove Geometrije} \label{osn9Geom}",
                                        "\poglavje{The basics of Geometry} \label{osn9Geom}")
                    ):
    prompt = f'''Translate only the text from the following LaTeX document into {dest_language}. Leave all LaTeX commands unchanged

"""
{sample_translation[0]}
{chunk}"""

{sample_translation[1]}
'''
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=1500,
    )
    result = response.choices[0].message.content.strip()
    result = result.replace('"""', '')  # remove the double quotes, as we used them to surround the text
    return result

def run():
    # OpenAI GPT-2 tokenizer is the same as GPT-3 tokenizer
    # we use it to count the number of tokens in the text
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # 1. Read in the data
    with open("data/geometry_slovenian.tex", "r") as f:
        text = f.read()

    # 1.1 Count the tokens in each chunk
    chunks = text.split('\n\n')
    ntokens = []
    for chunk in chunks:
        ntokens.append(len(tokenizer.encode(chunk)))
    print(max(ntokens))

    chunks = group_chunks(chunks, ntokens)
    print(len(chunks))

    dest_language = "English"

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(str(i + 1) + " / " + str(len(chunks)))
        # translate each chunk
        translated_chunks.append(translate_chunk(chunk, model='gpt-3.5-turbo', dest_language=dest_language))

    # join the chunks together
    result = '\n\n'.join(translated_chunks)

    # save the final result
    with open(f"data/geometry_{dest_language}.tex", "w") as f:
        f.write(result)

