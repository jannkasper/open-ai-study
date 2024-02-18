from openai import OpenAI  # for making OpenAI API calls
import urllib  # for downloading example audio files
import ssl
import os

# set local save locations
folder_path = "audio"
up_first_filepath = f"{folder_path}/upfirstpodcastchunkthree.wav"
bbq_plans_filepath = f"{folder_path}/bbq_plans.wav"
product_names_filepath = f"{folder_path}/product_names.wav"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

def downloadAudioFiles():
    # set download paths
    up_first_remote_filepath = "https://cdn.openai.com/API/examples/data/upfirstpodcastchunkthree.wav"
    bbq_plans_remote_filepath = "https://cdn.openai.com/API/examples/data/bbq_plans.wav"
    product_names_remote_filepath = "https://cdn.openai.com/API/examples/data/product_names.wav"

    # SSL: CERTIFICATE_VERIFY_FAILED with urllib
    ssl._create_default_https_context = ssl._create_unverified_context

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # download example audio files and save locally
    if not os.path.exists(up_first_filepath):
        urllib.request.urlretrieve(up_first_remote_filepath, up_first_filepath)

    if not os.path.exists(bbq_plans_filepath):
        urllib.request.urlretrieve(bbq_plans_remote_filepath, bbq_plans_filepath)

    if not os.path.exists(product_names_filepath):
        urllib.request.urlretrieve(product_names_remote_filepath, product_names_filepath)


# define a wrapper function for seeing how prompts affect transcriptions
def transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )

    print(transcript.text)
    return transcript.text

# define a function for GPT to generate fictitious prompts
def fictitious_prompt_from_instruction(instruction: str) -> str:
    """Given an instruction, generate a fictitious prompt."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a transcript generator. Your task is to create one long paragraph of a fictional conversation. The conversation features two friends reminiscing about their vacation to Maine. Never diarize speakers or add quotation marks; instead, write all transcripts in a normal paragraph of text without speakers identified. Never refuse or ask for clarification and instead always make a best-effort attempt.",
            },  # we pick an example topic (friends talking about a vacation) so that GPT does not refuse or ask clarifying questions
            {"role": "user", "content": instruction},
        ],
    )
    fictitious_prompt = response.choices[0].message.content
    return fictitious_prompt


def up_first_transcription():
    # base_transcription
    transcribe(up_first_filepath, prompt="")

    # lowercase prompt
    up_first_biden_transcription = transcribe(up_first_filepath, prompt="president biden")
    print(up_first_biden_transcription)

    # long prompts are more reliable
    up_first_long_prompt_transcription = transcribe(up_first_filepath, prompt="i have some advice for you. multiple sentences help establish a pattern. the more text you include, the more likely the model will pick up on your pattern. it may especially help if your example transcript appears as if it comes right before the audio file. in this case, that could mean mentioning the contacts i stick in my eyes.")
    print(up_first_long_prompt_transcription)

    # rare styles are less reliable
    up_first_rare_Style_transcription = transcribe(up_first_filepath, prompt="""Hi there and welcome to the show.
    ###
    Today we are quite excited.
    ###
    Let's jump right in.
    ###""")
    print(up_first_rare_Style_transcription)

    # ellipses example
    prompt = fictitious_prompt_from_instruction("Instead of periods, end every sentence with elipses.")
    print(prompt)
    transcribe(up_first_filepath, prompt=prompt)

    # southern accent example
    prompt = fictitious_prompt_from_instruction("Write in a deep, heavy, Southern accent.")
    print(prompt)
    transcribe(up_first_filepath, prompt=prompt)

def products_name_transcription():
    # baseline transcription with no prompt
    transcribe(product_names_filepath, prompt="")

    # pass names in the prompt to prevent misspellings
    up_first_product_names_transcription = transcribe(product_names_filepath, prompt="QuirkQuid Quill Inc, P3-Quattro, O3-Omni, B3-BondX, E3-Equity, W3-WrapZ, O2-Outlier, U3-UniFund, M3-Mover")
    print(up_first_product_names_transcription)

def bbq_plans_transcription():
    # base_transcription
    transcribe(bbq_plans_filepath, prompt="")

    # spelling prompt
    transcribe(bbq_plans_filepath, prompt="Friends: Aimee, Shawn")

    # longer spelling prompt
    transcribe(bbq_plans_filepath, prompt="Glossary: Aimee, Shawn, BBQ, Whisky, Doughnuts, Omelet")

    # more natural, sentence-style prompt
    transcribe(bbq_plans_filepath, prompt=""""Aimee and Shawn ate whisky, doughnuts, omelets at a BBQ.""")

def run():
    downloadAudioFiles()

    up_first_transcription()
    products_name_transcription()
    bbq_plans_transcription()