import os

path, dirs, files = next(os.walk(r"C:\Users\Admin\Desktop\waste"))
file_count = len(files)
print(file_count)
