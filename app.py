import os
import sys
import getopt
from orthomosaic import ImageMosaic


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def main():
    percentage = 20
    threshold = 0.9
    argument_list = sys.argv[2:]
    options = "pt:"
    long_options = ["Percent", "Threshold"]
    if len(sys.argv) < 2:
        print("Not enough arguments!")
        return

    try:
        arguments, values = getopt.getopt(argument_list, options, long_options)

        for current_arg, current_val in arguments:
            if current_arg in ("-p", "--Percent"):
                percentage = int(current_val)
            elif current_arg in ("-t", "--Threshold"):
                if float(current_val) < 1.00:
                    threshold = float(current_val)
    except getopt.GetoptError as err:
        print(str(err))

    path = str(sys.argv[1])
    # with cd("./Bird-Detectron-2"):
    #     os.system("python3 image_inference_bird.py -p " + path + " -m General_Model -t " + str(threshold))

    ImageMosaic.mosaic(path, percentage, threshold)

def weather():
    rl = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}"
    response = requests.get(url)
    data = response.json()

    # Extract the relevant weather information
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    description = data['weather'][0]['description']
    wind_speed = data['wind']['speed']
    # Add more weather parameters as needed

    # Print the weather information
    print(f"Weather information for {date} at {time}")
    print(f"Temperature: {temperature} K")
    print(f"Humidity: {humidity}%")
    print(f"Description: {description}")
    print(f"Wind Speed: {wind_speed} m/s")


if __name__ == "__main__":
    main()
