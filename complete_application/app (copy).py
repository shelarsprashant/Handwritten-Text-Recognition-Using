# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from main_code import input_image1


from read_user_img import user_input_path


from save_images_from_page2 import save_image_names_to_text_files


from  main_code2  import main1
# ==============================================================================================


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Handwritten Text Detection'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Handwritten Text Detection'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)
            file = request.files.get('file')

            print(file)
        #if not file:
         #   return render_template('disease.html', title=title)
        #try:
            img1 = file.read()
            with open('input.png', 'wb') as f:
                    f.write(img1)
            result="preprly saved the uploaded image"
            user_input_path()
            save_image_names_to_text_files()
            main_text=main1()


            return render_template('disease-result.html', prediction="The Text Extracted From Image is as Fallows: ",precaution=main_text,title="Handwritten Text Detection")
        #except:
         #   pass
    return render_template('disease.html', title=title)


# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
