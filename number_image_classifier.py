import cv2
import matplotlib.pyplot as plt
import os
from os import path

from mnist_classifier import *


MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")
data_folder = "./data/NumberImages/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define and load model
n_classes = 10
model = MNISTClassifier(n_classes).to(device)
model.load_state_dict(torch.load(MODEL_STATE_FILE))


for filename in os.listdir(data_folder):
    img = cv2.imread(data_folder + filename)
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in cnts:
        bounding_boxes.append(cv2.boundingRect(contour))
    bounding_boxes = sorted(bounding_boxes, key=lambda bb: bb[0])

    number_text = ""
    number_images = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        number_img = img[max(y-10, 0):min(y+h+10, img.shape[0]-1), max(x-10, 0):min(x+w+10, img.shape[1])]
        number_img = cv2.resize(number_img, (28, 28))
        number_images.append(number_img)

        number_input = torch.tensor(number_img).to(device)
        number_input = torch.transpose(number_input, 0, 1).unsqueeze(0).unsqueeze(0).float()

        output = model(number_input)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
        number_text += str(prediction)

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(number_text)

    plt.figure()
    for idx in range(len(number_images)):
        plt.subplot(1, len(number_images), idx+1)
        plt.imshow(number_images[idx], cmap='gray')
        plt.axis('off')

    plt.show()
