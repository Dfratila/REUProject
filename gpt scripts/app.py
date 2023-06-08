import os
import openai
import re

openai.api_key = ""


gpt_prompt = "only respond with a number; what is the sensor width of a DJI Mavic 2 Pro Drone with L1D-20c camera in millimeters"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {'role': 'user', 'content': gpt_prompt}
    ],
    temperature=0.0,
    max_tokens=5
)
regex = re.compile('[A-Za-z]')
num = float(regex.split(response.choices[0].message.content)[0])

print(response.choices[0].message.content)
print(num)

# response = openai.Completion.create(
#     engine="babbage",
#     prompt=gpt_prompt,
#     temperature=0.5,
#     max_tokens=5
# )
# print(response.choices[0].text)
