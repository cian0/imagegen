
#https://github.com/ageitgey/face_recognition

import face_recognition
import os 
import glob
from PIL import Image
import sys

_args = sys.argv[1:]
# Opens a image in RGB mode
 
# Cropped image of above dimension
# (It will not change original image)

orig_path = _args[0]
# orig_path = '/home/ian/Pictures/paularedygg/training_images/'
crop_outputs = _args[1]
# crop_outputs = '/home/ian/Pictures/paularedygg/cropped'

ctr = 1

images = glob.glob(f"{orig_path}/*")

if not os.path.exists(crop_outputs):
    os.makedirs(crop_outputs)

for image in images:
    
    print(image)
    loaded_img = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(loaded_img)
    im = Image.open(image)

    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size

    # (400, 225)
    # <class 'tuple'>

    w, h = im.size

    targetW = w
    targetH = h 
    if h > w:
        targetH = targetW
    else:
        targetW = targetH


    # Setting the points for cropped image
    if len(face_locations) == 0:
        print('cannot find face for ' + image)

        centerX = (targetW) / 2
        centerY = (targetH) / 2

        targetLeft = centerX - targetW / 2
        targetRight = centerX + targetW / 2
        targetTop = centerY - targetH / 2
        targetBottom = centerY + targetH / 2
        im1 = im.crop((targetLeft, targetTop, targetRight, targetBottom))


        im1.save(f"{crop_outputs}/cropped_{ctr}.png")
        ctr += 1 
        continue

    left = face_locations[0][0]
    top = face_locations[0][3]
    right = face_locations[0][2]
    bottom = face_locations[0][1]

    centerX = (left + right) / 2
    centerY = (top + bottom) / 2

    targetLeft = centerX - targetW / 2
    targetRight = centerX + targetW / 2
    targetTop = centerY - targetH / 2
    targetBottom = centerY + targetH / 2

    if h > w:
        # targetH = targetW
        targetLeft = 0
        targetRight = targetW
        if targetTop < 0:
            targetTop = 0
            targetBottom = targetH
        
        if targetBottom > h :
            targetTop = h - targetH
            targetBottom = h
    else:
        targetTop = 0
        targetBottom = targetH
        if targetLeft < 0:
            targetLeft = 0
            targetRight = targetW
        
        if targetRight > w :
            targetLeft = w - targetW
            targetRight = w

    im1 = im.crop((targetLeft, targetTop, targetRight, targetBottom))
    im1.save(f"{crop_outputs}/cropped_{ctr}.png")

    ctr += 1

    # with open(image, 'rb') as file:
    #     img = Image.open(file)

        # img.show()




# picture_of_me = face_recognition.load_image_file("me.jpg")
# my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

# unknown_picture = face_recognition.load_image_file("unknown.jpg")
# unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# # Now we can see the two face encodings are of the same person with `compare_faces`!

# results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

# if results[0] == True:
#     print("It's a picture of me!")
# else:
#     print("It's not a picture of me!")
