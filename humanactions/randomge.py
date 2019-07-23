import os
import random
import shutil

path =r'C:\Users\Admin\Desktop\waste1'
files = os.listdir(path)
dest=r'C:\Users\Admin\Desktop\waste'
for i in range(4):
    
    index = random.randrange(0, len(files))
    try:
        shutil.move(r'C:\Users\Admin\Desktop\waste1'+'\\'+files[index],dest)
    except:
        pass
