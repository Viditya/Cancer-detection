import gdown
from datetime import date 

date_today = date.today().strftime("%Y%m%d")

url = 'https://drive.google.com/uc?id=1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs'
# gdown.download(url, output, quiet=False)

output = 'CNN_assignment.zip'

file_dir = '/home/ubuntu/fun/lens/' + output

out_dir =  '/home/ubuntu/fun/lens/CNN_assignment'

print(file_dir)
print(out_dir)

import os
from zipfile import ZipFile

if os.path.exists(out_dir):
  print('File already zipped')
else:  
  os.makedirs(out_dir)
  # loading the temp.zip and creating a zip object
  with ZipFile(file_dir, 'r') as zObject:
  
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(path=out_dir)