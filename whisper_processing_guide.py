from openai import OpenAI
import os
import urllib
from IPython.display import Audio
from pathlib import Path
from pydub import AudioSegment
import ssl

# set local save locations
folder_path = "earnings_directory"
earnings_call_filepath = f"{folder_path}/EarningsCall.wav"
output_dir_trimmed = "trimmed_earnings_directory"  # Output directory for the segmented files

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

def downloadAudioFiles():
    # set download paths
    earnings_call_remote_filepath = "https://cdn.openai.com/API/examples/data/EarningsCall.wav"

    # SSL: CERTIFICATE_VERIFY_FAILED with urllib
    ssl._create_default_https_context = ssl._create_unverified_context

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # download example audio files and save locally
    if not os.path.exists(earnings_call_filepath):
        urllib.request.urlretrieve(earnings_call_remote_filepath, earnings_call_filepath)

# Function to detect leading silence
# Returns the number of milliseconds until the first sound (chunk averaging more than X decibels)
def milliseconds_until_sound(sound, silence_threshold_in_decibels=-20.0, chunk_size=10):
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold_in_decibels and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trim_start(filepath):
    path = Path(filepath)
    directory = path.parent
    filename = path.name
    audio = AudioSegment.from_file(filepath, format="wav")
    start_trim = milliseconds_until_sound(audio)
    trimmed = audio[start_trim:]
    new_filename = directory / f"trimmed_{filename}"
    trimmed.export(new_filename, format="wav")
    return trimmed, new_filename

def transcribe_audio(file,output_dir):
    audio_path = os.path.join(output_dir, file)
    with open(audio_path, 'rb') as audio_data:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_data)
        return transcription.text

# Define function to remove non-ascii characters
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)

# Define function to add punctuation
def punctuation_assistant(ascii_transcript):

    system_prompt = """You are a helpful assistant that adds punctuation to text.
      Preserve the original words and only insert necessary punctuation such as periods,
     commas, capialization, symbols like dollar sings or percentage signs, and formatting.
     Use only the context provided. If there is no context provided say, 'No context provided'\n"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response

# Define function to fix product mispellings
def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
     financial products and common financial terms are in the correct format. For each
     financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
     transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)'
    , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)'
, and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing 
financial products into their numeric representations, followed by the full name of the product in parentheses. 
For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'.
 However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 
'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to 
and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not 
represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to
 analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted 
 transcript and a list of the words you've changed"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response

def trimAudio():
    # Trim the start of the original audio file
    trimmed_audio, trimmed_filename = trim_start(earnings_call_filepath)

    # Segment audio
    trimmed_audio = AudioSegment.from_wav(trimmed_filename)  # Load the trimmed audio file

    one_minute = 1 * 60 * 1000  # Duration for each segment (in milliseconds)

    start_time = 0  # Start time for the first segment

    i = 0  # Index for naming the segmented files

    if not os.path.isdir(output_dir_trimmed):  # Create the output directory if it does not exist
        os.makedirs(output_dir_trimmed)

    while start_time < len(trimmed_audio):  # Loop over the trimmed audio file
        segment = trimmed_audio[start_time:start_time + one_minute]  # Extract a segment
        segment.export(os.path.join(output_dir_trimmed, f"trimmed_{i:02d}.wav"), format="wav")  # Save the segment
        start_time += one_minute  # Update the start time for the next segment
        i += 1  # Increment the index for naming the next file

def run():
    downloadAudioFiles()
    trimAudio()

    # Get list of trimmed and segmented audio files and sort them numerically
    audio_files = sorted(
        (f for f in os.listdir(output_dir_trimmed) if f.endswith(".wav")),
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    # Use a loop to apply the transcribe function to all audio files
    transcriptions = [transcribe_audio(file, output_dir_trimmed) for file in audio_files]

    # Concatenate the transcriptions
    print("Concatenate the transcriptions")
    full_transcript = ' '.join(transcriptions)
    print(full_transcript)

    # Remove non-ascii characters from the transcript
    print("Remove non-ascii characters from the transcript")
    ascii_transcript = remove_non_ascii(full_transcript)
    print(ascii_transcript)

    # Use punctuation assistant function
    print("Use punctuation assistant function")
    response = punctuation_assistant(ascii_transcript)
    # Extract the punctuated transcript from the model's response
    punctuated_transcript = response.choices[0].message.content
    print(punctuated_transcript)

    # Use product assistant function
    print("Use product assistant function")
    response = product_assistant(punctuated_transcript)
    # Extract the final transcript from the model's response
    final_transcript = response.choices[0].message.content
    print(final_transcript)










