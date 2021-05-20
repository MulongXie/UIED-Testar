from GUI import GUI
from Image import Image
import cv2
import glob

for g in glob.glob('data/dataset2/*.png'):
    name = g.split('.')[0]
    img_path = name + '.png'
    annot_path = name + '.json'
    print(name)

    gui = GUI(img_path, annot_path)
    gui.load_annotations()
    # gui.visualize_annotations(show_individual=False)

    gui.widget_detection()
    # gui.visualize_widgets(show_individual=False)

    gui.text_detection()
    gui.text_sentences_recognition()
    gui.text_shrink_bound()
    # gui.visualize_texts(line=2)

    # gui.visualize_all_elements()
    gui.merge_texts_widgets()
    board = gui.visualize_all_elements(show=False)

    cv2.imwrite('data/output/' + name.split('\\')[-1] + '.png', board)
    gui.save_all_elements('data/output/' + name.split('\\')[-1] + '.json')
    print('data/output/' + name.split('\\')[-1] + '.png\n')