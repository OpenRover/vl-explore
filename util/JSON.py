# JavaScript-like JSON API
import json

encoder = json.encoder.JSONEncoder(indent=None, separators=(",", ":"))
decoder = json.decoder.JSONDecoder()

def parse(string: str):
    return decoder.decode(string)

def stringify(obj):
    return encoder.encode(obj)
