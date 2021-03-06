import cv2
import numpy as np
from Element import Element


class Image:
    def __init__(self, img_file_name, resize_height=None):
        self.img_file_name = img_file_name
        self.img = cv2.imread(img_file_name)
        self.resize_height = resize_height
        self.resize_img_by_height()

        self.img_shape = self.img.shape
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]

        self.grey_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gradient_map = None
        self.binary_map = None
        self.get_gradient_map()
        self.get_binary_map()

        self.all_elements = []
        self.rectangle_elements = []
        self.square_elements = []
        self.line_elements = []

    '''
    ************************
    **** Pre-processing ****
    ************************
    '''
    def resize_img_by_height(self):
        if self.resize_height is not None:
            org_shape = self.img.shape
            resize_h = self.resize_height
            resize_w = int(org_shape[1] * (resize_h/org_shape[0]))
            self.img = cv2.resize(self.img, (resize_w, resize_h))

    def get_gradient_map(self):
        '''
        :return: gradient map
        '''
        img_f = np.copy(self.grey_img)
        img_f = img_f.astype("float")

        kernel_h = np.array([[0, 0, 0], [0, -1., 1.], [0, 0, 0]])
        kernel_v = np.array([[0, 0, 0], [0, -1., 0], [0, 1., 0]])
        dst1 = abs(cv2.filter2D(img_f, -1, kernel_h))
        dst2 = abs(cv2.filter2D(img_f, -1, kernel_v))
        gradient = (dst1 + dst2).astype('uint8')
        self.gradient_map = gradient
        return gradient

    def get_binary_map(self, min_grad=20, show=False):
        '''
        :param min_grad: if a pixel is bigger than this, then count it as foreground (255)
        :return: binary map
        '''
        rec, bin = cv2.threshold(self.gradient_map, min_grad, 255, cv2.THRESH_BINARY)
        morph = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, (3, 3))  # remove noises
        self.binary_map = morph
        if show:
            cv2.imshow('bin', morph)
            cv2.waitKey()
            cv2.destroyWindow('bin')
        return morph

    def get_binary_map_canny(self):
        self.binary_map = cv2.Canny(self.grey_img, 20, 20)
        return self.binary_map

    '''
    ***************************
    **** Element Detection ****
    ***************************
    '''
    def get_elements(self, min_area=10):
        '''
        get all elements on the image by findContours
        :return: list of [Component]
        '''
        self.all_elements = []
        _, contours, hierarchy = cv2.findContours(self.binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                ele = Element(contour=cnt)
                ele.get_clip(self.img)
                self.all_elements.append(ele)
        return self.all_elements

    def get_other_elements(self):
        others = []
        shapes = self.rectangle_elements + self.line_elements + self.square_elements
        for ele in self.all_elements:
            if ele not in shapes:
                others.append(ele)
        return others

    def detect_rectangle_and_square_elements(self, hollow=True):
        if len(self.all_elements) == 0:
            self.get_elements()
        self.rectangle_elements = []
        self.square_elements = []
        for ele in self.all_elements:
            rect_squ_check = ele.is_rectangle_or_square()
            if rect_squ_check:
                if hollow:
                    bin_clip = self.binary_map[ele.location['top']: ele.location['bottom'], ele.location['left']: ele.location['right']]
                    white_ratio = (np.sum(bin_clip) / 255) / (ele.width * ele.height)
                    # if too much white region, count as filled
                    if white_ratio > 0.5:
                        continue
                if rect_squ_check == 'square':
                    ele.type = 'square'
                    self.square_elements.append(ele)
                elif rect_squ_check == 'rectangle':
                    ele.type = 'rectangle'
                    self.rectangle_elements.append(ele)
        return self.rectangle_elements, self.square_elements

    def detect_line_elements(self):
        if len(self.all_elements) == 0:
            self.get_elements()
        self.line_elements = []
        for ele in self.all_elements:
            if ele.is_line():
                ele.type = 'line'
                self.line_elements.append(ele)
        return self.line_elements

    '''
    ***********************
    **** Visualization ****
    ***********************
    '''
    def visualize_binary_map(self):
        if self.binary_map is None:
            self.get_binary_map(show=True)
        else:
            cv2.imshow('binary', self.binary_map)
            cv2.waitKey()
            cv2.destroyWindow('binary')

    def visualize_elements_contours(self, board_opt='org', board=None, color=(255, 0, 0)):
        '''
        :param board_opt: 'org'/'binary'
        :return: drawn image
        '''
        contours = [ele.contour for ele in self.all_elements]
        if board is None:
            if board_opt == 'org':
                board = self.img.copy()
            elif board_opt == 'binary':
                board = np.zeros((self.img_shape[0], self.img_shape[1]))
        cv2.drawContours(board, contours, -1, color)
        cv2.imshow('contour', board)
        cv2.waitKey()
        cv2.destroyWindow('contour')
        return board

    def visualize_elements_contours_individual(self, board_opt='org', board=None, color=(255, 0, 0)):
        '''
        :param board_opt: 'org'/'binary'
        :param board: board image to draw on
        :return: drawn image
        '''
        if board is None:
            if board_opt == 'org':
                board = self.img.copy()
            elif board_opt == 'binary':
                board = np.zeros((self.img_shape[0], self.img_shape[1]))

        for ele in self.all_elements:
            cv2.drawContours(board, [ele.contour], -1, color)
            cv2.imshow('contour', board)
            cv2.waitKey()
        cv2.destroyWindow('contour')
        return board
