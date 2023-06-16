import os
import openai
import re

def sensor(model):
    try:
        openai.api_key_path = "../gpt_scripts/api-key"
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
