import json

with open('./src/resources/cache.json') as json_file:
    data = json.load(json_file)
    print(len(data.keys()))
