import cv2
import os
import requests
import json
from base64 import b64encode
import time


# ******** Google *********
def Google_OCR_makeImageData(imgpath):
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                # 'type': 'TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()


def ocr_detection_google(imgpath):
    start = time.clock()
    url = 'https://vision.googleapis.com/v1/images:annotate'
    api_key = 'AIzaSyDUc4iOUASJQYkVwSomIArTKhE2C6bHK8U'
    imgdata = Google_OCR_makeImageData(imgpath)
    response = requests.post(url,
                             data=imgdata,
                             params={'key': api_key},
                             headers={'Content_Type': 'application/json'})
    print('*** Text Detection Time Taken:%.3fs ***' % (time.clock() - start))
    if response.json()['responses'] == [{}]:
        # No Text
        return None
    else:
        return response.json()['responses'][0]['textAnnotations'][1:]
