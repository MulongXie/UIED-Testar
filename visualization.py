import cv2
import json
from glob import glob
from os.path import join as pjoin


def coord_transform(points):
    new_points = []
    for point in points:
        new_points.append((int(float(point['x']) * width),  int(float(point['y']) * height)))
    return new_points


def visualize_widget(widget_bound, board, show=True):
    cv2.rectangle(board, widget_bound[0], widget_bound[2], (0,0,255), 1)
    if show:
        cv2.imshow('widget', board)
        cv2.waitKey()
        cv2.destroyWindow('widget')


if __name__ == '__main__':
    json_files = glob('data\\dataset2\\' + '*.json')
    names = [n.split('\\')[-1][:-5] for n in json_files]

    js = json.load(open(json_files[0], 'r'))
    img = cv2.imread(pjoin('data', 'dataset2', js['screenshotId'] + '.png'))
    height, width = img.shape[:2]

    for widget in js['widgetJsonObjects']:
        #     board = img.copy()
        #     print(widget)
        new_bound = coord_transform(widget['boundingPoly']['stringVertice'])
        visualize_widget(new_bound, img, show=False)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyWindow('img')