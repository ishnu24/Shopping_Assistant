## SHOPPING ASSISTANT

Submitted for Lighthouse Labs final project.

Shopping_assistant.py does the following:
- Read images from the images folder
- It will call GoogleVision.py which will use the Google Vision API to locate all the logos an text in the images. and save it to a csv
- It will scan the images for produce using the produce.py, which in turn uses Test.py for object detection
- Will load shopping list (txt) and output of recommendaton engine (CSV)
- append recommendataions to shopping list as applicable
- Scan the results of the image analysis functions for objects in the shopping list
- Annotate the images accordinly and save them to a folder im_out.
