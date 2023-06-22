import os
import openai
import re
import tiktoken
import json
from collections import Counter


def sensor(model):
    try:
        openai.api_key_path = "./gpt_scripts/api-key"
        print(model)
        gpt_prompt = "only respond with a number; what is the sensor width of a " + model + " drone camera in millimeters"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'user', 'content': gpt_prompt}
            ],
            temperature=0.0,
            frequency_penalty=0.0,
            max_tokens=5
        )
    except openai.error.AuthenticationError or FileNotFoundError:
        print("API key missing!")
    else:
        regex = re.compile("[A-Za-z]")
        num = float(regex.split(response.choices[0].message.content)[0])
        print(num)
        return num


def analyze(path, confidence):
    try:
        openai.api_key_path = "./gpt_scripts/api-key"
        encoding = tiktoken.get_encoding("cl100k_base")
        with open(path) as file:
            birds = json.load(file)
            parse_json(birds, confidence)


    except openai.error.AuthenticationError or FileNotFoundError:
        print("API key missing!")
    else:
        return


def parse_json(file, confidence):
    count = Counter(k[:] for d in file for k, v in d.items() if k.startswith("confidence") and float(v) >= confidence)
    return int(count["confidence"])


if __name__ == "__main__":
    analyze("./orthomosaic/datasets/2_up/General_Model/bird.json", 0.9)
