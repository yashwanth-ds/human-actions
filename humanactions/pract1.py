import cv2


video_src = 'pedestrians.avi'

#cap = cv2.VideoCapture(video_src)
#c=r'C:\Users\Admin\Desktop\kth dataset\nikhil\boxing\walking_2398.jpg'
count=0
for e in range(2000,2440):
    e=r'C:\Users\Admin\Desktop\kth dataset\nikhil\boxing\walking_%d.jpg'%e
    img=cv2.imread(e)

    bike_cascade = cv2.CascadeClassifier('pedx_mcs.xml')

        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img)
    bike = bike_cascade.detectMultiScale(img,1.005,2)
    print(bike)
    if bike==():
        pass
    elif bike.all()>0:
        count=count+1
        #cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,210),4)
        
    cv2.imshow('video', img)
print(count)

    

#cv2.destroyAllWindows()
