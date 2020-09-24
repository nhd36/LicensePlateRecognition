from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def bounding_boxes(image):

	# pre-process the image by resizing it, converting it to
	# graycale, blurring it, and computing an edge map
	image = imutils.resize(image, height=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 50, 200, 255)

	# find contours in the edge map, then sort them by their
	# size in descending order
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	displayCnt = None
	# loop over the contours
	for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			# if the contour has four vertices, then we have found
			# the thermostat display
			if len(approx) == 4:
					displayCnt = approx
					break

	# extract the thermostat display, apply a perspective transform to it
	warped = four_point_transform(gray, displayCnt.reshape(4, 2))
	output = four_point_transform(image, displayCnt.reshape(4, 2))

	# threshold the warped image, then apply a series of morphological
	# operations to cleanup the thresholded image
	thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	# find contours in the thresholded image, then initialize the
	# digit contours lists
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# sort the contours from top to bottom, left to right
	cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * output.shape[1] )

	letters = {}
	n = 0
	for c in cnts:
		# extract the digit ROI
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
		letter = output[y:y+h, x:x+w]
		letters.update({"letter{0}".format(n):[x,y,letter]})
		n += 1

	return [letters, output]

def separate_lines(letters, output):
	# Threshold line divides the image into up_line and down_line
	threshold_line = output.shape[1]/3.5
	up_line = {}
	down_line = {}
	for key, value in letters.items():
			if letters[key][1] < threshold_line:
					up_line.update({value[0]:value[2]}) # Make x key and letter value
			elif letters[key][1] > threshold_line:
					down_line.update({value[0]:value[2]})

# Sort the order of the numbers on the plate
	up_line = {k:v for k,v in sorted(up_line.items(), key=lambda item: item[0])}
	down_line = {k:v for k,v in sorted(down_line.items(), key=lambda item: item[0])}
	return [up_line, down_line]

def predict(model, mapp, images, relabel=True):
    pred = []
    for image in images:
        HEIGHT = 28
        WIDTH = 12
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Reshape but keep the ratio of width and height as much as possible
        image = cv2.resize(image, (WIDTH, HEIGHT))

        image_arr = np.asarray(image)
        image_arr = image_arr.astype(np.float32)
        image_arr /= 255.0
        image_tolist = image_arr.tolist()

        pixels = []
        # Create a list of all pixels
        for pixel_list in image_tolist:
            for pixel in pixel_list:
                pixels.append(pixel)

        grayscale_letter = []
        threshold_black = 0.70
        # Change background to black, character to white
        for pixel in pixels:
            pixel = 1 - pixel
            if pixel < threshold_black:
                pixel = 0
            grayscale_letter.append(pixel)

        grayscale_letter = np.asarray(grayscale_letter)
        grayscale_letter = grayscale_letter.reshape(HEIGHT,WIDTH)

        image_input = grayscale_letter.reshape(1,28,12,1)
        label = model.predict_classes(image_input)

        if relabel == True:
            try:
                pred.append(mapp.iloc[:,0].values[label+1][0])
            except:
                pred.append(mapp.iloc[:,0].values[label-9][0])
        elif relabel == False:
            pred.append(mapp.iloc[:,0].values[label][0])

    return pred

def predict_license_plate(image_array):
    letters = bounding_boxes(image_array)[0]
    output = bounding_boxes(image_array)[1]
    up_line = separate_lines(letters, output)[0]
    down_line = separate_lines(letters, output)[1]

    # Load mapping file
    mapp_letters = pd.read_csv('models/label_char.csv',
        delimiter=',',
        index_col=0
        )

    mapp_plate = pd.read_csv('models/label_num.csv',
        delimiter=',',
        index_col=0
        )

    model_letters = load_model('models/model_char_plates_aug.h5')
    model_plate = load_model("models/model_num_plates_aug.h5")

    # Up line
    up_line_values = list(up_line.values())
    # Down line
    down_line_values = list(down_line.values())

    # Get rid of wrong bounding boxes
    up_line_values = [i for i in up_line_values if i.shape[0]*i.shape[1]>100]
    down_line_values = [i for i in down_line_values if i.shape[0]*i.shape[1]>100]


    # Numbers and letters in the up line
    up_line_nums = [up_line_values[0], up_line_values[1], up_line_values[4]]
    up_line_char = [up_line_values[3]]

    # Run two models on nums and letters
    pred_up_nums = predict(model_plate, mapp_plate, up_line_nums)
    pred_up_char = predict(model_letters, mapp_letters, up_line_char, relabel=False)
    upper_line = f"{pred_up_nums[0]}{pred_up_nums[1]}-{pred_up_char[0]}{pred_up_nums[2]}"

    # There are plates with 6 boxes and there are plates with 4 boxes in the down line
    if len(down_line_values) == 6:
        down_line_values.pop(3)
        pred_down_nums = predict(model_plate, mapp_plate, down_line_values)
        down_line = f"{pred_down_nums[0]}{pred_down_nums[1]}{pred_down_nums[2]}.{pred_down_nums[3]}{pred_down_nums[4]}"
    else:
        pred_down_nums = predict(model_plate, mapp_plate, down_line_values)
        down_line = f"{pred_down_nums[0]}{pred_down_nums[1]}{pred_down_nums[2]}{pred_down_nums[3]}"

    license_plate = upper_line + " | " + down_line
    return license_plate
