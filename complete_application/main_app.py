import numpy as np

from detect import main_function

import os
import shutil


from PIL import Image
def crop_words():
                    import numpy as np

                    from detect import main_function

                    import os
                    import shutil


                    from PIL import Image
                    # Open an image file (replace 'input.png' with the actual file name)
                    from PIL import Image
                    image_path = 'input.png'
                    img = Image.open(image_path)

                    # Convert the image to RGB mode
                    img = img.convert('RGB')

                    # Resize the image to 416x416
                    #new_size = (416, 416)
                    #img_resized = img.resize(new_size)

                    # Save the resized image as 'input_resized.jpg'
                    output_path = 'input.jpg'
                    #img_resized.save(output_path)
                    img.save(output_path)

                    #print(f"{image_path} resized to  and saved as {output_path}")





                    folder_path = "runs/detect"

                    # Check if the folder exists
                    if os.path.exists(folder_path):
                        # List all files and folders in the directory
                        files_and_folders = os.listdir(folder_path)

                        # Iterate through each item
                        for item in files_and_folders:
                            item_path = os.path.join(folder_path, item)

                            # Check if it's a file or a directory
                            if os.path.isfile(item_path):
                                # Delete the file
                                os.remove(item_path)
                                print(f"Deleted file: {item}")
                            elif os.path.isdir(item_path):
                                # Delete the directory and its contents
                                shutil.rmtree(item_path)
                                print(f"Deleted folder: {item}")
                    else:
                        print(f"The folder {folder_path} does not exist.")

                    from PIL import Image
                    import os



                    main_function()

