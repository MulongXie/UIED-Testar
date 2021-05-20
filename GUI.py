import cv2
import json
import os
import ocr
from Text import Text
from Image import Image


class GUI:
    def __init__(self, img_path, annotation_path=None):
        self.img_path = img_path
        self.img_obj = Image(img_path)
        self.img = self.img_obj.img
        self.annotation_path = annotation_path
        self.annotations = []

        self.texts = []
        self.widgets = []

        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

    def load_annotations(self):
        if self.annotation_path and os.path.exists(self.annotation_path):
            js = json.load(open(self.annotation_path, 'r'))
            for obj in js['widgetJsonObjects']:
                bound = []
                for point in obj['boundingPoly']['stringVertice']:
                    bound.append((int(float(point['x']) * self.width), int(float(point['y']) * self.height)))
                self.annotations.append(bound)
        else:
            print('No annotation file exits')

    '''
    ***************************
    **** Element Detection ****
    ***************************
    '''
    def widget_detection(self):
        elements = self.img_obj.get_elements()
        # remove noise
        for ele in elements:
            if ele.height/ele.width >= 5:
                ele.is_abandoned = True
        # remove overlapping
        for i in range(len(elements) - 1):
            ei = elements[i]
            if ei.is_abandoned: continue
            for j in range(i + 1, len(elements)):
                ej = elements[j]
                if ej.is_abandoned: continue
                # if redundant, remove the smaller one
                if ei.is_ele_overlay(ej):
                    if ei.area >= ej.area:
                        ej.is_abandoned = True
                    else:
                        ei.is_abandoned = True
        for ele in elements:
            if not ele.is_abandoned:
                self.widgets.append(ele)

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

    '''
    ************************
    **** Element Refine ****
    ************************
    '''
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

    def text_shrink_bound(self):
        for text in self.texts:
            text.shrink_bound(self.img_obj.binary_map)

    '''
    ***********************
    **** Visualization ****
    ***********************
    '''
    def get_img_copy(self):
        return self.img.copy()

    def visualize_widgets(self, color=(255, 0, 0), line=2, show=True, show_individual=False):
        board = self.get_img_copy()
        for widget in self.widgets:
            widget.visualize_element(board, color, line)
            if show_individual:
                cv2.imshow('widget', board)
                cv2.waitKey()
                cv2.destroyWindow('widget')
                board = self.img.copy()
        if show:
            cv2.imshow('widgets', board)
            cv2.waitKey()
            cv2.destroyWindow('widgets')

    def visualize_texts(self, color=(0, 255, 0), line=2, show=True, show_individual=False):
        board = self.get_img_copy()
        for text in self.texts:
            text.visualize_text(board, color, line)
            if show_individual:
                cv2.imshow('text', board)
                cv2.waitKey()
                cv2.destroyWindow('text')
                board = self.img.copy()
        if show:
            cv2.imshow('texts', board)
            cv2.waitKey()
            cv2.destroyWindow('texts')

    def visualize_annotations(self, color=(0, 0, 255), line=1, show=True, show_individual=False):
        board = self.get_img_copy()
        for annot in self.annotations:
            cv2.rectangle(board, annot[0], annot[2], color, line)
            if show_individual:
                cv2.imshow('annot', board)
                cv2.waitKey()
                cv2.destroyWindow('annot')
                board = self.img.copy()
        if show:
            cv2.imshow('annotations', board)
            cv2.waitKey()
            cv2.destroyWindow('annotations')
