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
    argument_list = sys.argv[2:]
    options = "p:"
    long_options = ["Percent"]
    if len(sys.argv) < 2:
        print("Not enough arguments!")
        return

    try:
        arguments, values = getopt.getopt(argument_list, options, long_options)

        for current_arg, current_val in arguments:
            if current_arg in ("-p", "--Percent"):
                percentage = int(current_val)
    except getopt.GetoptError as err:
        print(str(err))

    path = str(sys.argv[1])
    with cd("./Bird-Detectron-2"):
        os.system("python3 image_inference_bird.py -p " + path + " -m General_Model -t 0.9")

    ImageMosaic.mosaic(path, percentage)

if __name__ == "__main__":
    main()
