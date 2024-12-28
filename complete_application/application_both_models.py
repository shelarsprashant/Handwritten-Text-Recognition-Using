import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from main_app import crop_words
import tensorflow as tf
def all_task():
                                crop_words()
                                tf.compat.v1.reset_default_graph()

                                with open('img_names_sequence.txt',"r") as f:
                                    list_imgs = f.readlines()


                                grades=[]

                                for i in range(len(list_imgs)):
                                    grades.append(list_imgs[i].strip('\n'))


                                print(list_imgs)

                                result_sentence=[]


                                class FilePaths:
                                    """Filenames and paths to data."""
                                    fn_char_list = 'model/charList.txt'
                                    fn_summary = 'model/summary.json'
                                    fn_corpus = 'data/corpus.txt'


                                def get_img_height() -> int:
                                    """Fixed height for NN."""
                                    return 32


                                def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
                                    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
                                    if line_mode:
                                        return 256, get_img_height()
                                    return 128, get_img_height()







                                def infer(model: Model, fn_img: Path) -> None:
                                    """Recognizes text in image provided by file path."""
                                    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
                                    assert img is not None

                                    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
                                    img = preprocessor.process_img(img)

                                    batch = Batch([img], None, 1)
                                    recognized, probability = model.infer_batch(batch, True)
                                    print(f'Recognized: "{recognized[0]}"')
                                    print(f'Probability: {probability[0]}')
                                    return recognized[0]


                                #def prediction_each_img(img_name_input):

                                def main1():
                                    """Main function."""
                                    parser = argparse.ArgumentParser()

                                    parser.add_argument('--mode', choices=['infer'], default='infer')
                                    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
                                    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
                                    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
                                    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
                                    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
                                    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='data/word.png')
                                    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
                                    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')
                                    args = parser.parse_args()

                                    # set chosen CTC decoder
                                    decoder_mapping = {'bestpath': DecoderType.BestPath,
                                                       'beamsearch': DecoderType.BeamSearch,
                                                       'wordbeamsearch': DecoderType.WordBeamSearch}
                                    decoder_type = decoder_mapping[args.decoder]

                                    # train or validate on IAM dataset
                                    if args.mode in ['train', 'validate']:
                                        print("not needed")

                                    elif args.mode == 'infer':
                                        model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True, dump=args.dump)

                                        for xyz in grades:
                                               print("till this ok")
                                               try:

                                                      resullt_word_pred=infer(model,xyz)
                                                      result_sentence.append(resullt_word_pred)
                                                      print("the final sentence is ",result_sentence)
                                               except:
                                                      print("the final sentence is ",result_sentence)
                                    print(' '.join(str(x) for x in result_sentence))
                                    return result_sentence
                                    #finall_data_converted_to_text_from_image=(' '.join(str(x) for x in result_sentence))

                                    #file_text_output= open("text_output_for_scanned_input_img.txt","w")
                                    #file_text_output.close()

                                    #file_text_output.write(finall_data_converted_to_text_from_image)
                                    #tf.compat.v1.reset_default_graph()
                                    #return result_sentence
                                    




                                p=main1()
                                return  p

