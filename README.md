The models can't be stored so they have to be downloaded seperately
https://drive.google.com/drive/folders/1Cv41H6agrxOlDDcfSmHs_4AgAINCwVQT

To use the app, add a file named 'api-key' to the gpt_scripts folder and paste your openai key inside.

To run: `python3 app.py <full folder pathname> -p <downscale percentage (optional)> -t <confidence threshold (optional>`

Folder pathname should be the absolute pathname
Downscale percentage is default 20%, if the images are too big, the orthomosaic generation will not work and the process will be killed
Confidence threshold default is 0.90
