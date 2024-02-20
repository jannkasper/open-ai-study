# imports
from openai import OpenAI  # for making OpenAI API calls
import urllib  # for downloading example audio files
import os  # for accessing environment variables
import ssl

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

# set download paths
ZyntriQix_remote_filepath = "https://cdn.openai.com/API/examples/data/ZyntriQix.wav"

# set local save locations
folder_path = "audio"
ZyntriQix_filepath = f"{folder_path}/ZyntriQix.wav"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# download example audio files and save locally
if not os.path.exists(ZyntriQix_filepath):
    # SSL: CERTIFICATE_VERIFY_FAILED with urllib
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(ZyntriQix_remote_filepath, ZyntriQix_filepath)

# define a wrapper function for seeing how prompts affect transcriptions
def transcribe(prompt: str, audio_filepath) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    print(transcript.text)
    return transcript.text

# define a wrapper function for seeing how prompts affect transcriptions
def transcribe_with_spellcheck(system_message, audio_filepath):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": transcribe(prompt="", audio_filepath=audio_filepath),
            },
        ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def run():
    # baseline transcription with no prompt
    transcribe(prompt="", audio_filepath=ZyntriQix_filepath)

    # add the correct spelling names to the prompt
    transcribe(
        prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T.",
        audio_filepath=ZyntriQix_filepath,
    )

    # add a full product list to the prompt
    transcribe(
        prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, AstroPixel Array, QuantumFlare Five, CyberPulse Six, VortexDrive Matrix, PhotonLink Ten, TriCircuit Array, PentaSync Seven, UltraWave Eight, QuantumVertex Nine, HyperHelix X, DigiSpiral Z, PentaQuark Eleven, TetraCube Twelve, GigaPhase Thirteen, EchoNeuron Fourteen, FusionPulse V15, MetaQuark Sixteen, InfiniCircuit Seventeen, TeraPulse Eighteen, ExoMatrix Nineteen, OrbiSync Twenty, QuantumHelix TwentyOne, NanoPhase TwentyTwo, TeraFractal TwentyThree, PentaHelix TwentyFour, ExoCircuit TwentyFive, HyperQuark TwentySix, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T.",
        audio_filepath=ZyntriQix_filepath,
    )

    system_prompt = "You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."
    transcribe_with_spellcheck(system_prompt, audio_filepath=ZyntriQix_filepath)

    system_prompt = "You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array,  OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, AstroPixel Array, QuantumFlare Five, CyberPulse Six, VortexDrive Matrix, PhotonLink Ten, TriCircuit Array, PentaSync Seven, UltraWave Eight, QuantumVertex Nine, HyperHelix X, DigiSpiral Z, PentaQuark Eleven, TetraCube Twelve, GigaPhase Thirteen, EchoNeuron Fourteen, FusionPulse V15, MetaQuark Sixteen, InfiniCircuit Seventeen, TeraPulse Eighteen, ExoMatrix Nineteen, OrbiSync Twenty, QuantumHelix TwentyOne, NanoPhase TwentyTwo, TeraFractal TwentyThree, PentaHelix TwentyFour, ExoCircuit TwentyFive, HyperQuark TwentySix, GigaLink TwentySeven, FusionMatrix TwentyEight, InfiniFractal TwentyNine, MetaSync Thirty, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."
    transcribe_with_spellcheck(system_prompt, audio_filepath=ZyntriQix_filepath)

