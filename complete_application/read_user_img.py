
# Python program to read image using OpenCV
 # Python program to read image using OpenCV
  
# importing OpenCV(cv2) module
import cv2
  

def user_input_path(): 
    # Read RGB image
    img = cv2.imread('input.png')

    # Save image in a specified directory
    cv2.imwrite('data/page/example.png', img) 

# Call the function to execute
