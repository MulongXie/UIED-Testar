import cv2
import os
import ocr
from Text import Text


class GUI:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.texts = []
        self.widgets = []

        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

    def text_detection(self):
        ocr_result = ocr.ocr_detection_google(self.img_path)
        if ocr_result is not None:
            for result in ocr_result:
                x_coordinates = []
                y_coordinates = []
                text_location = result['boundingPoly']['vertices']
                text = result['description']
                for loc in text_location:
                    x_coordinates.append(loc['x'])
                    y_coordinates.append(loc['y'])
                location = {'left': min(x_coordinates), 'top': min(y_coordinates),
                            'right': max(x_coordinates), 'bottom': max(y_coordinates)}
                self.texts.append(Text(text, location))

    def text_sentences_recognition(self, bias_justify=3, bias_gap=15):
        '''
        Merge separate words detected by Google ocr into a sentence
        '''
        changed = True
        while changed:
            changed = False
            temp_set = []
            for text_a in self.texts:
                merged = False
                for text_b in temp_set:
                    if text_a.is_on_same_line(text_b, 'h', bias_justify=bias_justify, bias_gap=bias_gap):
                        text_b.merge_text(text_a)
                        merged = True
                        changed = True
                        break
                if not merged:
                    temp_set.append(text_a)
            self.texts = temp_set.copy()

    def visualize_texts(self, color=(0, 255, 0), line=1, show=True):
        board = self.img.copy()
        for text in self.texts:
            text.visualize_text(board, color, line)
        if show:
            cv2.imshow('texts', board)
            cv2.waitKey()
            cv2.destroyWindow('texts')
