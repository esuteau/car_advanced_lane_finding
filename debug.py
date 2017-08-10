import cv2
import matplotlib.pyplot as plt

def PlotRoadImageExample():
    image = 'test_images/straight_lines2.jpg'

    # Read image and convert
    img_bgr = cv2.imread(image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    f, ax1 = plt.subplots(1, 1, figsize=(20, 10))
    ax1.imshow(img_rgb)
    plt.show()    


PlotRoadImageExample()