import argparse
from typing import List

import cv2
import matplotlib.pyplot as plt
from path import Path
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt1

from word_detector import detect, prepare_img, sort_multiline

#from main_code import main1

import numpy as np
from PIL import Image as im



list_img_names_serial=[]


def get_img_files(data_dir: Path) -> List[Path]: 
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)

        print(res)
    return res



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path('data/page'))
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--sigma', type=float, default=11)
parser.add_argument('--theta', type=float, default=5)
parser.add_argument('--min_area', type=int, default=100)
parser.add_argument('--img_height', type=int, default=1000)
parsed = parser.parse_args()
def save_image_names_to_text_files():
        for fn_img in get_img_files(parsed.data):
            print(f'Processing file {fn_img}')

            # load image and process it
            img = prepare_img(cv2.imread(fn_img), parsed.img_height)
            detections = detect(img,
                                kernel_size=parsed.kernel_size,
                                sigma=parsed.sigma,
                                theta=parsed.theta,
                                min_area=parsed.min_area)

            # sort detections: cluster into lines, then sort each line
            lines = sort_multiline(detections)

           # print("the sorted lines  of the each images is ",lines)
            import os

            folder_path12 = "segmented_images"

            # Check if the folder exists
            if os.path.exists(folder_path12):
                # Get a list of all files in the folder
                files = os.listdir(folder_path12)
                
                # Iterate over each file and remove it
                for file in files:
                    file_path = os.path.join(folder_path12, file)
                    os.remove(file_path)
                print("Contents of the folder 'segmented_images' have been removed.")
            else:
                print(f"The folder '{folder_path12}' does not exist.")

            # plot results
            #plt.imshow(img,cmap='gray')
            #im2=plt.imshow(img,cmap='gray')
            num_colors = 7
            colors = plt.cm.get_cmap('rainbow', num_colors)
            for line_idx, line in enumerate(lines):
                print("line index and line is printed",line_idx)
                #list_img_names_serial.clear()
                for word_idx, det in enumerate(line):
                    print("word index in line is printed",word_idx)
                    #print("image array is det",det)
                    xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                    ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                    #plt.plot(xs, ys, c=colors(line_idx % num_colors))
                    #plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
                    #print(det.bbox.x,det.bbox.y,det.bbox.w,det.bbox.h)
                    crop_img = img[det.bbox.y:det.bbox.y+det.bbox.h,det.bbox.x:det.bbox.x+det.bbox.w]
                    cv2.imwrite("segmented_images/"+"line"+str(line_idx)+"word"+str(word_idx)+".jpg",crop_img)
                    full_img_path="segmented_images/"+"line"+str(line_idx)+"word"+str(word_idx)+".jpg"
                    list_img_names_serial.append(full_img_path)
                    print(list_img_names_serial)
                    #list_img_names_serial_set= set(list_img_names_serial)


                    #print("image path as input",full_img_path)
                    #resultent_sentence=main1(full_img_path)
                    #print("full sentence from the images",resultent_sentence)


                    #croped =im2(xs,ys)
                   # plt.imshow(img,cmap='gray')
                    #plt1.imshow(croped,cmap='gray')
                    #plt1.plot(xs, ys, c=colors(line_idx % num_colors))
                    #plt1.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

                    #plt1.savefig('foo.png')


                textfile = open("img_names_sequence.txt", "w")
            #textfile.truncate(0)
                for element in list_img_names_serial:
                      textfile.write(element + "\n")
                textfile.close()


                   

           # plt.show()




