import os
import cv2
import sys
import json
import datetime
import tempfile
import requests
import numpy as np
from PIL import Image

subscription_key = "235e4a6177ca4359a64bcc69f0bbe49a"
assert subscription_key
vision_base_url = "https://southeastasia.api.cognitive.microsoft.com/vision/v2.0/recognizeText?"


def resize(image):
    imgW, imgH = image.shape[:2]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image_path = temp_file.name
    image = cv2.resize(image, (imgH//2, imgW//2))
    cv2.imwrite(image_path, image)
    return image_path


def tiff2img(imagePath):
    image = Image.open(imagePath)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tempImagePath = temp_file.name
    image.save(tempImagePath, 'JPEG', quality=96)
    return tempImagePath


def azure(img):
    image_data = cv2.imencode('.jpg', img)[1].tostring()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'mode': 'Printed'}
    response = requests.post(
        vision_base_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]
    analysis = {}
    poll = True
    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        if ("recognitionResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'Failed'):
            poll = False
    polygons = []
    result = []
    intx, inty, finx, finy = 0, 0, 0, 0
    if ("recognitionResult" in analysis):
        word = [(line["boundingBox"], line["text"]) for line in analysis["recognitionResult"]["lines"]]
    return word


def helpercode(image_path):
    filename, file_extension = os.path.splitext(image_path)
    if file_extension in (".TIF", ".tif", ".TIFF", ".tiff"):
        image_path = tiff2img(image_path)
        image = cv2.imread(image_path)
    elif file_extension in (".jpg", ".jpeg", ".Jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        image = cv2.imread(image_path)

    temp_path = ""
    try:
        imgW, imgH = image.shape[:2]
        while(imgW > 4000 or imgH > 4000):
            temp_path = resize(image)
            image = cv2.imread(temp_path)
            imgW, imgH = image.shape[:2]
    except:
        pass
    if temp_path != "":
        image = cv2.imread(temp_path)

    word = azure(image)

    extracted_text = ""
    for i in range(0, len(word)):
        extracted_text += word[i][1] + " "

    return extracted_text.strip()


#if __name__ == "__main__":
#    image_path = "images.jpeg"
#    extracted_text = extract_text_from_image(image_path)
#    print("Extracted Text:")
#    print(extracted_text)
#image_path = "images.jpeg"
#extracted_text = extract_text_from_image(image_path)
#print(extracted_text)