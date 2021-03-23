# Import numpy and pandas
import numpy as np
import pandas as pd
# import pillow for image annotation
import PIL
from PIL import Image, ImageDraw, ImageFont
# import os for readingg and writing to files
import os

from scipy.spatial.distance import cdist
# import custom files and functions
import produce
import GoogleVision

# Folder to locate images
BASEPATH="images"

# Run the Google vision funtions to analyze the images and output a csv
# Uncomment below when required
#GoogleVision.main(BASEPATH)

# Send produce images to produce function and store in DF
df_produce=pd.DataFrame()
basepath = "images"
#df_produce = df_produce.append(produce.main(basepath, 'produce-on-grocery-shelf.jpg'))
df_produce = df_produce.append(produce.main(basepath, 'shoppers-drug-mart-produce-fresh-fruit.jpg', 8))

# Import table of text from images, convert all text to lowercase
df_txt = pd.read_csv('text.csv', index_col=0)
df_txt['description']=df_txt['description'].str.lower()

# Import table of logos from images, convert all text to lowercase
df_logo = pd.read_csv('logo.csv', index_col=0)
df_logo['description']=df_logo['description'].str.lower()

# Import the shopping listdir
df_list = pd.read_csv('Shopping List.txt', names=['items', 'categories','recommended'])
df_list['recommended']=0

# import recommendation tabe, created by usign apriori on a grocery dataset
df_recomended = pd.read_csv('recommendation_table.csv')

# Import sample recommendations from recommendations.txt
df_recs=pd.DataFrame()
df_temp=pd.read_csv('recommendations.txt', names=['categories','item_2', 'recommended']).drop(columns=['categories'], axis=1)

# Loop through items in the shopping list to find what items are likely to be bought with it
# append these items to the recommendation list.
for r in df_list['categories']:
    #print(df_recomended[df_recomended['item_1'] == r])
    y = df_recomended[df_recomended['item_1'] == r].sort_values(by='confidence', ascending=False).head(2)
    df_temp=df_temp.append(y)

# add the recomended items to the shopping list.
# Set the recommended flag to 1 to indicate it was a recomendation and not on
#   the oringinal shopping list
df_recs['items']=df_temp['item_2']
df_recs['recommended']=1
df_recs = df_recs.reset_index().drop(columns=['index'])
df_list = df_list.append(df_recs, ignore_index=True)


def label_items(i, y, df, df_x, source, recommend):
    """
    i: item from the shopping list
    y: the index of that item in the DF
    df: the main DF of this project that holds all the image coordinates
    df_x: the logo, text, or produce DF from the API calls
    source: indicates which API it came from
    recommend: flag to indicate if it was from the shopping or recommendation list

    This Function takes an item name(i), df index(y), main DF to append to (df),
    dataframe with the text and coordinates from google API df_x,
    identifier to indicate which API it used (source),
    flag for original list or recommendation engine (recommend)
    """
    line={}   # Create an empyt Dict
    x=df_x.iloc[y]      # assign the row at y
    line['item'] = i    # assign i to the item of the Dict
    line['description'] = str(x['description'])     # assign the the description from df_x to the dict
    line['pic'] = x['pic']  # add pic (the file name of the image) to the Dict
    line['x0'] = x['x0'] - 20   # assign the coordinates of the bounding box
    line['y0'] = x['y0'] - 20   #   but add 20px to all sides
    line['x1'] = x['x1'] + 20
    line['y1'] = x['y1'] + 20
    line['source'] = source     # add source to the dict
    line['rec'] = recommend     # add the recommend flag  to the Dict
    df = df.append(line, ignore_index=True)     # append the dict to the main DF
    return df




def remain(df, df_list, df_txt):
    """
    Loop through all the remaining items in the shopping list.
    split each item into it's individual words, and search the txt DF
    for each word.  Send the words, item by item to the 'search' function.
    It will return a bounding box that estimates the object.

    """
    # loop through items in shopping list
    for _,r in df_list.iterrows():
        i=r['items']
        recommend=r['recommended']
        df_temp=pd.DataFrame()
        locations = {}
        max_char=len(i)
        words = i.lower().split()
        words_len = len(words)
    #    print(words_len,words)
        # for each word of an item in the grocery list.
        for j in words:
            word=[]
            pic = []
            x0=[]
            y0=[]
            x1=[]
            y1=[]
            a=[]
            rec=[]
            #for index, rows in (df_txt[[j in x for x in df_txt['description'].astype(str)]]).iterrows():
            for index, rows in df_txt[df_txt['description'].astype(str).str.contains(j)].iterrows():
                if len(rows['description']) < max_char:
                    #print(j, rows['description'])
                    word.append(rows['description'])
                    pic.append(rows['pic'])
                    x0.append(rows['x0'])
                    y0.append(rows['y0'])
                    x1.append(rows['x1'])
                    y1.append(rows['y1'])
                    a.append((rows['x0'],rows['y0']))
                    rec.append(recommend)
                    locations={'word':word, 'pic':pic,'x0':x0,'y0':y0,'x1':x1,'y1':y1, 'rec':rec,'a':a}

                #print(locations)
                    df_temp=df_temp.append(pd.DataFrame(locations), ignore_index=True)
        x0, y0, x1, y1 =search(df_temp, words)
        df_temp['x0']=x0
        df_temp['y0']=y0
        df_temp['x1']=x1
        df_temp['y1']=y1
        df_temp['item']=i
        df=df.append(df_temp, ignore_index=True)
    return df





def search(df, words):
    """
    gets a df with all the words in the item, located in an image.
    using scipy cdist, to find the two words in item that are closest together.
    it will create a bounding box around those two words, as an estimate of the
    object.


    """
    prod={}
    tables={}
    pairs=[]
#    print(words)
    word_len = len(words)
#    print(word_len)
    if df.empty:
        # if df is empty for this item, skip
        return(-100,-100,-100,-100)
    else:
        # loop through each word in the item name
        # place all the coordinate pairs 'a' for each word into a dict 'prod'
        for i in range(len(words)):
            k=i
#            print(i)
            prod[i]=df[df['word']==words[i]]['a'].values.tolist()
            if not prod[i]:
                # if the word wasn't found in the text remove it from
                #    the list of words and continue
                words.remove(words[i])
                i=i-1
                if not words:
                    return(-100,-100,-100,-100)
#                print(i)
#            print(i)

        # loop through each word pair using prod i and i+1
        for i in range(len(words)-1):
            # save the distances for each set of coordinate pairs into table
            tables[i]=cdist(prod[i], prod[i+1])
            # Save the smallest value for each pair of words as pairs
            pairs.append(np.min(tables[i]))
#            print('\n',words[i], words[i+1])
#            print('min: ',np.min(tables[i]))
#            print('argmin: ',np.argmin(tables[i]))

        # the index of the smallest value is the pair of words closest together
        # we assume they are on the same package
        pair = np.argmin(pairs) # element of table index that holds the min value
        shape=tables[pair].shape
        # pos is the row and column that holds the min value
        pos = np.unravel_index(tables[pair].argmin(),tables[pair].shape)

#        print(np.argmin(tables[pair]),tables[pair].shape, pos)
#        print(df.iloc[pos[0]])
#        print(df.iloc[(shape[0]+pos[1])])

        #df_temp=df.iloc[pos[0]]
        x0 = min(df.iloc[(shape[0]+pos[1])]['x1'],df.iloc[pos[0]]['x0'])
        y0 = min(df.iloc[(shape[0]+pos[1])]['y1'],df.iloc[pos[0]]['y0'])
        x1 = max(df.iloc[(shape[0]+pos[1])]['x1'],df.iloc[pos[0]]['x0'])
        y1 = max(df.iloc[(shape[0]+pos[1])]['y1'],df.iloc[pos[0]]['y0'])
#        print(df_temp)


        return(x0, y0, x1, y1)

def display_image(df):
    """
    Takes the main DF, and places a label on the image for each item in the
    df.  the colour of the text and background is based on the recommend flag.
    """
    # loop through the DF
    for filename in df['pic']:
        #print(filename)

        # Load an image
        basepath = "images"
        pic = os.path.join(basepath, filename)
        frame = df[df['pic']==filename]


        # Get the size of the image to scale the labels as a percentage of
        #   the image size
        with Image.open(pic) as im:
            width, height = im.size

            # If the image is small, have a minimum size for the labels
            height = max(height, 550)
            width = max(width, 550)
            border = int(height * 0.05)
            box_h = int(height * 0.01)
            box_w = int(width * 0.01)
            # This is the font to use, size is scaled as above
            font = ImageFont.truetype('BebasNeue-Regular.ttf', border)

            # Loop through the DF for each image
            for index, row, in frame.iterrows():
                # assign text and coordinates from the DF to variables
                name = row['item']
                x0 = row['x0'] - box_w
                y0 = row['y0'] - box_h
                x1 = row['x1'] + box_w
                y1 = row['y1'] + box_h

                # Assign colours based on the recommend flag
                if row['rec']:
                    text_color = 'green'
                    box_color = 'green'
                else:
                    text_color = 'red'
                    box_color = 'red'



                # Set the image type to RGBA to allow for opacity settings
                draw = ImageDraw.Draw(im, 'RGBA')

                #draw.rectangle([(x0,y0),(x1,y1)], fill=(100, 100, 0, 20), outline=box_color, width=int(border*.2))
                #draw.rectangle([(x0,y0),(x1,y1)], fill=None, outline='aqua', width=int(border *.1))
                #draw.rectangle([(x0,y0),(x1,y1)], fill=None, outline=box_color, width=int(border* .05))


                # Get the size of the text to base the background text box on
                xb0, yb0, xb1, yb1 = draw.textbbox((x0,y0), str(name) ,anchor='ld', font = font, align='left')

                # If any part of the text is off the image, shift the whole
                #   whole text box into the image
                if xb1 > width:
                    xb0=xb0-(xb1-(width- box_w))
                    xb1=width-box_w
                elif xb0 < 0:
                    xb1 = xb1-xb0 + 3*(box_w)
                    xb0 = box_w
                else:
                    xb1=xb1+box_w
                    xb0=xb0-box_w

                if yb0 < 0:
                    yb1 = yb1-yb0+ 3*(box_h)
                    yb0 = box_h
                elif yb1 > height:
                    yb0=yb0-(yb1-(height- box_h))
                    yb1=height-box_h
                else:
                    yb1=yb1+box_h
                    yb0=yb0-box_h

                # Draw the text box and the text in the location provided by the DF
                draw.rectangle([(xb0,yb0),(xb1,yb1)], fill=box_color, outline='yellow', width=int(border*.05))
                draw.text((xb0+box_w,yb0), str(name),anchor='la', font = font, align='left', fill='black', outline='white')


            #save the annotated image to a folder
            im.save(os.path.join('im_out', filename), 'jpeg')


df = pd.DataFrame(columns=['item', 'description', 'pic','x0','y0','x1','y1','source','rec'])



def main_loop(df_main, df_txt, df_logo, df_active):

    # loop through items in shopping list
    for _,r in df_active.iterrows():
        i=r['items']
        recommend=r['recommended']
        # If the items are in the logo DF, populate main DF with information
        x = df_logo[df_logo['description']== i.lower()]
        if not x.empty:
            #print(x.index)
            for j in x.index:
                # Send the found item to the lable function
                df_main = label_items(i, j, df_main, df_logo, 'logo', recommend)
                # Remove found items from the shopping list
                df_active = df_active[df_active['items']!=i]

                # issues with pasta sauce, remove it until bug is fixed
                df_active = df_active[df_active['items']!='pasta sauce']


        # If the item is in the produce DF, populate main DF with information
        x = df_produce[df_produce['description']==i.lower()]
        if not x.empty:
            for j in x.index:
                df_main = label_items(i, j, df_main, df_produce, 'produce', recommend)
                df_active = df_active[df_active['items']!=i]

        # If the items aren't in the logo DF, check the text DF
        x = df_txt[df_txt['description']== i.lower()]
        if not x.empty:
            #print(x.index)
            for j in x.index:
                df_main = label_items(i, j, df_main, df_txt, 'text', recommend)
                df_active = df_active[df_active['items']!=i]

    # Break down the text for the remaining items and search the text DF
    df_main = remain(df_main, df_active, df_txt)
#    print(df_active['items'])
    return df_main


# send everything to the main funtion to run
df = main_loop(df, df_txt, df_logo, df_list)

# remove duplicated items
df.drop_duplicates(subset=['item'], inplace=True)
#print(df)
display_image(df)
print('done!')
#df
