import numpy as np
import cv2
import random

triangle=0
square=0
pentagon=0
circle=0
rectangle=0
cocoa=0
noncocoa=0
a=0
b=0
bcocoa=[]
bshape=[]
#Cocoa Detect
for i in range(1,11):
    image = cv2.imread("biscuits/"+str(i)+".jpg")

# define the list of boundaries
    boundaries = [
    	([0, 0, 0], [100, 100, 100])
    ]
    
    # loop over the boundaries
    for (lower, upper) in boundaries:
    	# create NumPy arrays from the boundaries
    	lower = np.array(lower, dtype = "uint8")
    	upper = np.array(upper, dtype = "uint8")
    
    	# find the colors within the specified boundaries and apply
    	# the mask
       
    	mask = cv2.inRange(image, lower, upper)
    	output = cv2.bitwise_and(image, image, mask = mask)
    
    
    #cv2.imshow("images", np.hstack([image, output]))
    meanoutput=np.mean(output)
    if meanoutput<1:
        noncocoa+=1
        a=0
    elif meanoutput>=1:
        cocoa+=1
        a=1
    bcocoa.append(a)
print ('Total Cocoa Biscuits :',cocoa)
print('Total Non-cocoa Biscuits : ',noncocoa)


#Shape Detect
for i in range(1,11):
    img = cv2.imread("biscuits/"+str(i)+".jpg",1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3),0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
    contours,_ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    filtered = []

    objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')
    
    for c in contours:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.04*peri,True)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
           
            if ar >= 0.95 and ar <= 1.05:
                 shape = "square"
                 
            else:
                shape="rectangle"
                
    
    		# if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
            
    
    		# otherwise, we assume the shape is a circle
        else:
            shape = "circle"
            
    
    if shape=='triangle':
        triangle+=1
        b=1
    elif shape=='square':
        square+=1
        b=2
    elif shape=='pentagon':
        pentagon+=1
        b=3
    elif shape=='circle':
        circle+=1
        b=4
    elif shape=='rectangle':
        rectangle+=1
        b=5
    bshape.append(b)
print ('Triangle Count : ',triangle)
print ('Square Count : ',square)
print ('Pentagon Count : ',pentagon)
print ('Circle Count : ',circle)
print ('Rectangle Count : ',rectangle)
print(bcocoa)
print(bshape)

import csv
with open('deneme.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Cocoa", "Shape"])
    for i in range(0,10):
         writer.writerow([bcocoa[i], bshape[i]])
        
   
cv2.waitKey(0)
cv2.destroyAllWindows()
        