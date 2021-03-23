
import os
import pandas as pd

def detect_logos(path):
    """Detects logos in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.logo_detection(image=image)
    logos = response.logo_annotations
#    print('Logos:')


    for logo in logos:
#        print(logo.description)
        pass

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return(response, logos)


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
#    print('Texts:')
        

    for text in texts:
#        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

#        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return(response, texts)


def main(basepath="images"):
    import os
    import pandas as pd

    df_txt = pd.DataFrame(columns=['pic', 'description','x0','y0','x1','y1'])
    df_logo = pd.DataFrame(columns=['pic', 'description','x0','y0','x1','y1'])
#    basepath = "images"

    for filename in os.listdir(basepath):
#        print(os.path.join(basepath, filename))
        pic = os.path.join(basepath, filename)

        response, texts = detect_text(pic)

        for text in texts:
            line = {}
            line['pic']=filename
            line['description']=text.description
            line['x0'] = text.bounding_poly.vertices[0].x
            line['y0'] = text.bounding_poly.vertices[0].y
            line['x1'] = text.bounding_poly.vertices[2].x
            line['y1'] = text.bounding_poly.vertices[2].y
            df_txt = df_txt.append(line, ignore_index=True)


        response, logos = detect_logos(pic)

        for logo in logos:
            line = {}
            line['pic']=filename
            line['description']=logo.description
            line['x0'] = logo.bounding_poly.vertices[0].x
            line['y0'] = logo.bounding_poly.vertices[0].y
            line['x1'] = logo.bounding_poly.vertices[2].x
            line['y1'] = logo.bounding_poly.vertices[2].y
            df_logo = df_logo.append(line, ignore_index=True)


    print('API responses returned')
    df_txt.to_csv('text.csv')
    df_logo.to_csv('logo.csv')
