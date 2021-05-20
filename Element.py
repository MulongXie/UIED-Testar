import cv2


class Element:
    def __init__(self, id=None, type=None, contour=None, location=None, clip_img=None):
        self.id = id
        self.type = type
        self.contour = contour          # format of findContours
        self.location = location        # dictionary {left, right, top, bottom}
        self.clip_img = clip_img

        self.is_abandoned = False       # if the element is noise

        self.width = None
        self.height = None
        self.area = None
        self.init_bound()

    def init_bound(self):
        if self.location is not None:
            self.width = self.location['right'] - self.location['left']
            self.height = self.location['bottom'] - self.location['top']
            self.area = self.width * self.height
        else:
            self.get_bound_from_contour()

    def get_bound_from_contour(self):
        if self.contour is not None:
            bound = cv2.boundingRect(self .contour)
            self.width = bound[2]
            self.height = bound[3]
            self.area = self.width * self.height
            self.location = {'left': bound[0], 'top': bound[1], 'right': bound[0] + bound[2], 'bottom': bound[1] + bound[3]}

    def get_clip(self, org_img):
        self.clip_img = org_img[self.location['top']: self.location['bottom'], self.location['left']: self.location['right']]

    '''
    *******************************
    *** Relation with Other Ele ***
    *******************************
    '''
    def calc_intersection(self, element, bias=0, board=None):
        '''
        :return: ioa(self) and iob(the element)
        '''
        l_a = self.location
        l_b = element.location
        left_in = max(l_a['left'], l_b['left']) + bias
        top_in = max(l_a['top'], l_b['top']) + bias
        right_in = min(l_a['right'], l_b['right'])
        bottom_in = min(l_a['bottom'], l_b['bottom'])

        w_in = max(0, right_in - left_in)
        h_in = max(0, bottom_in - top_in)
        area_in = w_in * h_in
        ioa = area_in / self.area
        iob = area_in / element.area

        if board is not None and ioa != 0:
            print('ioa:%.3f; iob:%.3f' % (ioa, iob))
            element.visualize_element(board)
            self.visualize_element(board, show=True)
        return ioa, iob

    def pos_relation(self, element, bias=0, board=None):
        '''
        Calculate the relation between two elements by iou
        :return:
        -1  : a in b
         0  : a, b are not intersected
         1  : b in a
         2  : a, b are intersected
        '''
        ioa, iob = self.calc_intersection(element, bias, board)
        # area of intersection is 0
        if ioa == 0:
            return 0
        # a in b
        if ioa > 0.6:
            return -1
        # b in a
        if iob == 1:
            return 1
        return 2

    def is_ele_overlay(self, ele_b, point_bias=3):
        loc_a = self.location
        loc_b = ele_b.location
        if abs(loc_a['top'] - loc_b['top']) <= point_bias and abs(loc_a['bottom'] - loc_b['bottom']) <= point_bias and \
                abs(loc_a['left'] - loc_b['left']) <= point_bias and abs(loc_a['right'] - loc_b['right']) <= point_bias:
            return True
        ioa, iob = self.calc_intersection(ele_b)
        if max(ioa, iob) == 1 and min(ioa, iob) >= 0.9:
            return True
        return False

    def is_overlay_or_contained(self, ele_b, point_bias=3):
        loc_a = self.location
        loc_b = ele_b.location
        if abs(loc_a['top'] - loc_b['top']) <= point_bias and abs(loc_a['bottom'] - loc_b['bottom']) <= point_bias and \
                abs(loc_a['left'] - loc_b['left']) <= point_bias and abs(loc_a['right'] - loc_b['right']) <= point_bias:
            return True
        ioa, iob = self.calc_intersection(ele_b)
        if ioa > 0.7:
            return True
        return False

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def visualize_element(self, image, color=(255, 0, 0), line=2, show=False):
        cv2.rectangle(image, (self.location['left'], self.location['top']), (self.location['right'], self.location['bottom']), color, line)
        if show:
            cv2.imshow('element', image)
            cv2.waitKey()
            cv2.destroyWindow('element')
