from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import azure.cognitiveservices.speech as speechsdk
import os
from pydub import AudioSegment
from dotenv import load_dotenv
import openai
from pydub.silence import split_on_silence
import threading

from src.tokopediaScraper import Tokopedia
from src.response import Response

app = FastAPI(title="HETI",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load env variable
load_dotenv()
SPEECH_KEY = os.getenv('SPEECH_KEY')
SPEECH_REGIONS = os.getenv('SPEECH_REGIONS')
OPENAI_TOKEN = os.getenv('OPENAI_TOKEN')
CAT_AMOUNT = os.getenv('CAT_AMOUNT')


def getRecommendation(text: str) -> list:
    openai.api_key = OPENAI_TOKEN

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": "You are a very informative Doctor who gives informational responses"},
        {"role": "user",
         "content": '"saya merasa pusing dan sering kehilangan keseimbangan". Berikan saya alat kesehatan yang dapat membantu saya. Respond in this format: "Item: [item], Item: [item]". Example: "Item: tensimeter, Item: masker". Dont explain anything.'},
        {"role": "user", "content": f"{text}. Berikan saya {CAT_AMOUNT} alat."},
    ], temperature=0, max_tokens=1000)

    content = response["choices"][0]["message"]["content"]
    items = [i.replace("Item:", "").strip() for i in content.split(",")]

    return items


def threadGetProducts(products, tool: str):
    tokopedia = Tokopedia("src/resources/chromedriver/chromedriver.exe")

    produk = tokopedia.search(tool)
    products.append(produk)

    tokopedia.close_connection()


def getProducts(tools: list):
    try:
        products = []
        threads = []

        for tool in tools:
            t = threading.Thread(target=threadGetProducts,
                                 args=(products, tool))
            t.start()
            threads.append(t)

        for thread in threads:
            thread.join()

        return products
    except Exception as e:
        return HTTPException(status_code=400, detail=e)


@app.post("/api/upload")
def upload_audio(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        filename = ".\\upload_files\\"+file.filename

        with open(filename, 'wb') as f:
            f.write(contents)

        # Webm to Wav Converter
        audio = AudioSegment.from_file(filename, format="webm")
        os.remove(filename)
        filename = filename[:-4]+"wav"
        audio.export(filename, format="wav")

        sound = AudioSegment.from_file(filename, format="wav")
        audio_chunks = split_on_silence(
            sound, min_silence_len=1, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in audio_chunks:
            combined += chunk
        os.remove(filename)
        combined.export(filename, format="wav")

        # Azure STT API
        speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY, region=SPEECH_REGIONS)
        audio_config = speechsdk.audio.AudioConfig(filename=filename)

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, language="id-ID", audio_config=audio_config)

        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.NoMatch:
            raise HTTPException(status_code=400, detail="Speech no match")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            return HTTPException(status_code=400, detail=cancellation_details)

        tools = getRecommendation(result.text)

        return Response(success=True, data=getProducts(tools))

    except Exception:
        return HTTPException(status_code=400, detail="Failed")
    finally:
        file.file.close()


@app.get("/api/recommendation")
def recommendations(query: str):
    try:
        tools = getRecommendation(query)

        return Response(success=True, data=getProducts(tools))

    except Exception:
        return HTTPException(status_code=400, detail="Failed")
