
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

np.random.seed(42)
tf.random.set_seed(42)



base_path = "data"
words_list = []


# In[3]:




words = open(f"{base_path}/words.txt", "r").readlines()



#print(words[0:10])
for line in words:
    if line[0] == "#":
        continue
    if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
        words_list.append(line)

len(words_list)

np.random.shuffle(words_list)




split_idx = int(0.9 * len(words_list))



# In[8]:


train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]

print(train_samples[0],test_samples[0])


# In[9]:


val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]


assert len(words_list) == len(train_samples) + len(validation_samples) + len(
    test_samples
)




# In[12]:


base_image_path = os.path.join(base_path, "words")


# In[13]:


def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        #print(line_split)
        line_split = line_split.split(" ")
        #print(line_split)

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
      #  print(image_name)
        partI = image_name.split("-")[0]
      #  print(partI)
        partII = image_name.split("-")[1]
       # print("part 2",partII)
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])
           # print(corrected_samples[0])

    return paths, corrected_samples


# In[14]:


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)



# In[17]:


# Find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)


# Check some label samples.
train_labels_cleaned[:10]


# In[18]:


#import pickle
 
#with open('characters.pkl', 'wb') as f:

#   pickle.dump(characters, f)


# In[19]:


import pickle
with open('characters.pkl', 'rb') as f:
   characters = pickle.load(f)




import pickle
 
#with open('vocab1.pkl', 'wb') as f:

#   pickle.dump(vocab1, f)

import pickle
with open('vocab1.pkl', 'rb') as f:
   vocab1= pickle.load(f)




def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned = clean_labels(test_labels)


# In[24]:


test_labels_cleaned[0:10]


# In[25]:


AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=vocab1, mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


#num_to_char 


# In[27]:


#example for string lookup
#https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup

#vocab = ["a", "b", "c", "d"]
#data = tf.constant([["a", "c", "d"], ["m", "z", "b"]])
#layer = tf.keras.layers.StringLookup(vocabulary=vocab, num_oov_indices=2)
#layer(data)


# In[28]:


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


# In[29]:


batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


# In[30]:





# In[32]:


#https://www.tensorflow.org/api_docs/python/tf/data/Dataset

#example   for     tf.data.Dataset.from_tensor_slices
#dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
#for element1 in dataset1:
#  print(element1)


# In[47]:


batch_size = 32
padding_token = 99
image_width = 128
image_height = 32


def preprocess_image(image_path, img_size=(image_width, image_height)):     #called 4th 
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)    #calling above function  here
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))              #called 3rd
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):                     #called 2nd
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}






def prepare_dataset(image_paths, labels):           #called 1
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(         
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


# In[48]:


train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
test_ds = prepare_dataset(test_img_paths, test_labels_cleaned)


# In[49]:


#for data in train_ds.take(1):
#    images, labels = data["image"], data["label"]
#    print("this is image in tensor",images)
#    print("this is the tensor value for the label",label)


# In[50]:



# In[62]:


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model():
    # Inputs to the model
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((image_width // 4), (image_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    return model


# Get the model.
model = build_model()
model.summary()


# In[63]:


validation_images = []
validation_labels = []

for batch in validation_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])


# In[64]:



# In[65]:


#https://www.tensorflow.org/api_docs/python/tf/edit_distance


def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :max_len]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )


# In[66]:


epochs =50 # To get good results this should be at least 50.

model = build_model()
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
edit_distance_callback = EditDistanceCallback(prediction_model)

#model.compile()

#try:
#             prediction_model.save("handwritten_text.h5")

#             prediction_model.save_weights("ckpt")
#             load_status = sequential_model.load_weights("ckpt")

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
#             load_status.assert_consumed()
#             model1 = keras.models.load_model('handwritten_text.h5')

#except:
    
#             print("models are not saved and loaded")
    


# In[ ]:





# In[ ]:





# In[57]:


# Train the model.
#history = model.fit(
#    train_ds,
#    validation_data=validation_ds,
#    epochs=epochs
#)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


#prediction_model.save("handwritten_text_50.h5")


# In[59]:



model1 = keras.models.load_model('handwritten_text_50.h5')

#model1 =tf.saved_model.load(('handwritten_text.h5'))


# In[60]:


# A utility function to decode the output of the network.
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def   prediction_on_user_input(img):

                        test_img_paths[0]=img
                        #print(len(test_img_paths))
                        test_ds1 = prepare_dataset(test_img_paths, test_labels_cleaned)


                        #  Let's check results on some test samples.
                        for batch in test_ds1.take(1):
                            #print(batch)
                            batch_images = batch["image"]
                            _, ax = plt.subplots(5, 5, figsize=(15, 8))

                            preds = model1.predict(batch_images)
                            #print(preds)
                            pred_texts = decode_batch_predictions(preds)
                            print(pred_texts)

                            for i in range(25):
                               # img = batch_images[i]
                               # img = tf.image.flip_left_right(img)
                               # img = tf.transpose(img, perm=[1, 0, 2])
                               # img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
                                #img = img[:, :, 0]

                                title = f"Prediction: {pred_texts[i]}"
                               # ax[i // 5, i % 5].imshow(img, cmap="gray")
                               # ax[i // 5, i % 5].set_title(title)
                               # ax[i // 5, i % 5].axis("off")
                        return pred_texts[0]
                        #plt.show()


# In[ ]:


#print(prediction_on_user_input("p5.png"))


# In[2]:


#import nltk
#nltk.edit_distance("amladipina", "amlodipine")


# In[4]:


#import difflib

#a = 'amladipina'
#b = 'amlodipine'

#seq = difflib.SequenceMatcher(None,a,b)
#d = seq.ratio()*100
#print(d) 


# In[ ]:




