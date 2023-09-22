from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import azure.cognitiveservices.speech as speechsdk
import http.client
import os
import json
import re
from pydub import AudioSegment
from dotenv import load_dotenv
import openai
from pydub.silence import split_on_silence
import threading
from keras.models import load_model
import pandas as pd
from src.service.tokopediaScraper import Tokopedia
from src.service.lazadaScraper import Lazada
from src.service.response import Response
import uuid

ECOMMERCE = ["tokopedia", "lazada"]
cache_file_path = "src/resources/cache.json"


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
RAJAONGKIR_KEY = os.getenv('RAJAONGKIR_KEY')

# Setup
model = load_model('src/resources/ranking')
openai.api_key = OPENAI_TOKEN

# ====================================================================================================
# Service function


def ranker(data):
    def encode_price_category(price):
        if price <= 50000:
            return 0
        elif price <= 100000:
            return 1
        elif price <= 300000:
            return 2
        elif price <= 1000000:
            return 3
        else:
            return 4

    def sort_result(result):
        # Associate each element with its index
        indexed_result = list(enumerate(result))
        # Sort based on values
        sorted_result = sorted(indexed_result, key=lambda x: x[1])
        return sorted_result

    data.loc[:, 'price'] = data['price'].apply(encode_price_category)

    result = model.predict(data)
    return sort_result(result)


def cek_alkes(text: str) -> bool:
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": 'You are a doctor who is helpful to the user'},
        {"role": "user", "content": f'"{text}"\nDoes this statement suggest that the person needs medical help?\nRespond "Yes" or "No" only'}
    ], temperature=0.7, max_tokens=1000)

    answer = response["choices"][0]["message"]["content"]
    isAlkes = 'yes' in answer.lower()

    return isAlkes


def getRecommendation(text: str) -> list:

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        {"role": "system", "content": "You are a very informative Doctor who gives informational responses"},
        {"role": "user",
         "content": '"saya merasa pusing dan sering kehilangan keseimbangan". Berikan saya alat kesehatan yang dapat membantu saya. Respond in this format: "Item: [item], Item: [item]". Example: "Item: tensimeter, Item: masker". Dont explain anything.'},
        {"role": "user", "content": f"{text}. Berikan saya {CAT_AMOUNT} alat."},
    ], temperature=0, max_tokens=1000)

    content = response["choices"][0]["message"]["content"]
    items = [i.replace("Item:", "").strip() for i in content.split(",")]

    for idx in range(len(items)):
        items[idx] = re.sub(r'[^\w\s]', '', items[idx])

    return items


def threadGetProducts(products, ECOMMERCE: str, tool: str):

    if ECOMMERCE == "tokopedia":
        tokopedia = Tokopedia()
        produk = tokopedia.search(tool)
        tokopedia.close_connection()
    elif ECOMMERCE == "lazada":
        lazada = Lazada()
        produk = lazada.search(tool)
        lazada.close_connection()

    try:
        products[tool] += produk
    except Exception as e:
        return


def getProducts(tools: list):
    try:
        products = dict()
        threads = []

        cache = dict()
        with open(cache_file_path, 'r') as outfile:
            cache = json.load(outfile)

        for tool in tools:
            products[tool] = []

            if tool in cache:
                products[tool] += cache[tool]
                continue

            for e in ECOMMERCE:

                t = threading.Thread(target=threadGetProducts,
                                     args=(products, e, tool))
                t.start()
                threads.append(t)

        for thread in threads:
            thread.join()

        res = []
        for tool in tools:
            try:
                produk = pd.DataFrame(products[tool])

                produk = produk.dropna()

                ranks = ranker(produk[['price', 'rating', 'sold']])
                ranks = [id[0] for id in ranks]
                produk = produk.reindex(ranks)
                produk = produk.drop("rank", axis=1).reset_index(
                    drop=True).reset_index()
                produk = produk.rename(columns={"index": "rank"})
                produk["rank"] = produk["rank"].apply(lambda x: x+1)
                produk = produk.dropna()
                produk = produk.to_dict(orient="records")

                cache[tool] = produk

                res.append(produk)
            except Exception as e:
                cache[tool] = products[tool]
                res.append(products[tool])
                continue

        new_cache = dict()
        with open(cache_file_path, 'r') as outfile:
            new_cache = json.load(outfile)

        for tool in tools:
            new_cache[tool] = cache[tool]

        with open(cache_file_path, "w") as outfile:
            json.dump(new_cache, outfile)

        return res
    except Exception as e:
        return HTTPException(status_code=400, detail=e)


# ====================================================================================================
# API ENDPOINT


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        filename = ".\\upload_files\\"+str(uuid.uuid4())+"-"+file.filename

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

        if (not cek_alkes(result.text)):
            return Response(success=False, code=400, message="Sepertinya query yang anda masukkan tidak terkait dengan alat kesehatan.", data=[])

        tools = getRecommendation(result.text)

        data = dict()
        data["message"] = result.text
        data["products"] = getProducts(tools)

        return Response(success=True, data=data)

    except Exception as e:
        return HTTPException(status_code=400, detail=e)
    finally:
        file.file.close()


@app.get("/api/recommendation")
async def recommendations(query: str):
    try:
        if (not cek_alkes(query)):
            return Response(success=False, code=400, message="Sepertinya query yang anda masukkan tidak terkait dengan alat kesehatan.", data=[])
        tools = getRecommendation(query)

        data = dict()
        data["message"] = query
        data["products"] = getProducts(tools)

        return Response(success=True, data=data)

    except Exception as e:
        return HTTPException(status_code=400, detail=e)


@app.get("/api/cekongkir")
async def cek_ongkir(src: str, dest: str, weight=1000):
    try:
        conn = http.client.HTTPSConnection("api.rajaongkir.com")

        payload = f"origin={src}&destination={dest}&weight={weight}&courier=jne"

        headers = {
            'key': RAJAONGKIR_KEY,
            'content-type': "application/x-www-form-urlencoded"
        }

        conn.request("POST", "/starter/cost", payload, headers)

        res = conn.getresponse()
        data = json.loads(res.read())

        costs = 0
        if "rajaongkir" in data and "results" in data["rajaongkir"]:
            results = data["rajaongkir"]["results"]
            for result in results:
                if "costs" in result:
                    for cost in result["costs"]:
                        if "cost" in cost and isinstance(cost["cost"], list):
                            for value in cost["cost"]:
                                if "value" in value:
                                    costs += value["value"]

        return Response(success=True, data=costs/2)

    except Exception as e:
        return HTTPException(status_code=400, detail=e)


@app.get("/api/")
async def hello_world():
    return Response(success=True, data={'message': "Hello World!"})
