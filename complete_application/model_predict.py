#import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
loaded_model_imageNet = load_model("model_resnet50.h5")
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import io
from PIL import Image
def pred_leaf_disease(image_path):				 
				# Opens a image in RGB mode
			#	transform = transforms.Compose([
				#transforms.Resize(256),
				#transforms.ToTensor(),
			#	])
				#image = Image.open(io.BytesIO(image_path))
				image = Image.open(image_path).convert("RGB")


				newsize = (100,100)
				img = image.resize(newsize)



				x = np.expand_dims(img, axis=0)
				x = preprocess_input(x)
				result = loaded_model_imageNet.predict(x)
				print((result*100).astype('int'))
				final_list_result=(result*100).astype('int')
				list_vals=list(final_list_result[0])
				result_val=max(list(final_list_result[0]))
				print(result_val)
				index_result = list_vals.index(result_val)
				return   index_result

#print(pred_leaf_disease('corn.JPG'))