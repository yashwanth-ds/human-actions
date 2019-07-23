import cv2


video_src = 'pedestrians.avi'

#cap = cv2.VideoCapture(video_src)
#c=r'C:\Users\Admin\Desktop\goa\P1020914.jpg'
e=r'C:\Users\Admin\Downloads\Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master\Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master\Main Project\Main Project\Pedestrian Detection\frame25.jpg'
img=cv2.imread(e)

bike_cascade = cv2.CascadeClassifier('pedx_mcs.xml')

    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray)
#print(img)
bike = bike_cascade.detectMultiScale(img,1.05,2)
print(bike)
x=0

for(a,b,c,d) in bike:
    print('human detected')
    #cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,210),4)
    
cv2.imshow('video', img)
#print(x)

    

#cv2.destroyAllWindows()
