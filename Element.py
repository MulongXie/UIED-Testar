import cv2


class Element:
    def __init__(self, id=None, type=None, contour=None, location=None, clip_img=None):
        self.id = id
        self.type = type
        self.contour = contour          # format of findContours
        self.location = location        # dictionary {left, right, top, bottom}
        self.clip_img = clip_img

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

    def visualize_element(self, image, color=(255, 0, 0), line=2, show=False):
        cv2.rectangle(image, (self.location['left'], self.location['top']), (self.location['right'], self.location['bottom']), color, line)
        if show:
            cv2.imshow('element', image)
            cv2.waitKey()
            cv2.destroyWindow('element')
