import Test
import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Divide image into sections and scan each section for produce
#   return a DF with produce and coordinates

def main(basepath, filename, PIECES=5):
    # set up the functions from Test.py
    print(filename)
    execution_path = os.getcwd()
    prediction = Test.CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("model_ex-010_acc-0.997730.h5")
    prediction.setJsonPath("model_class.json")
    prediction.loadModel(num_objects=32)

    df_produce = pd.DataFrame(columns=['description','pic','x0','y0','x1','y1'])
    #PIECES = 5    # number of columns and rows to divide the image into
    pic = os.path.join(basepath, filename)

    #  Open image and determine
    with Image.open(pic) as im:
        width, height = im.size
        box_h = int(height / PIECES)
        box_w = int(width / PIECES)
        # loop across image, drop down to next row and loop across again
        for y in range(PIECES):
            y0 = y*box_h
            y1 = y0+box_h
            for x in range(PIECES):
                item={}
                x0 = x*box_w
                x1 = x0+box_w
                # crop box size
                box=(x0, y0, x1, y1)

                # crop image and send selection to Test.py prediction function
                a=im.crop(box)
                items,probs=prediction.predictImage(a, input_type='array', result_count=2)

                # If probability is over 95%, save name of produce and filename
                #    and coordinates.  Append to dataframe and return.
                if float(probs[0]) > 95:
                    item['description']=items[0].split()[0].lower()+'s'
                    item['pic']=filename
                    item['x0'] = x0
                    item['y0'] = y0
                    item['x1'] = x1
                    item['y1'] = y1
                    df_produce =df_produce.append(item, ignore_index=True)

    return df_produce
