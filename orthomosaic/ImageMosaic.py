'''
Driver script. Execute this to perform the mosaic procedure.
'''

import utilities as util
import Combiner
import cv2

fileName = "./datasets/imageData.txt"
imageDirectory = "./datasets/01052022/"
allImages, image = util.importData(imageDirectory)
print(image)
# stitcher = cv2.Stitcher_create()
#
# #stitched = allImages[0]
# #for i in range(1, len(allImages)):
#     #(status, stitched) = stitcher.stitch([stitched, allImages[i]])
# (status, stitched) = stitcher.stitch(allImages)
# if status == 0:
#     # cv2.imwrite("results/finalResult.png", stitched)
#     # cv2.imshow("Result", stitched)
#     # cv2.waitKey(0)
# else:
#     print("Failed status:" + str(status))
'''
myCombiner = Combiner.Combiner(allImages)
result = myCombiner.createMosaic()
util.display("RESULT", result)
cv2.imwrite("results/finalResult.png", result)
'''

















