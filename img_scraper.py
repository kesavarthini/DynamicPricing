import numpy as np
import pandas as   pd
import csv
import requests

Output = "/home/swarnalatha/Desktop/pics"
file = pd.read_csv("/home/swarnalatha/Desktop/final_train.csv")
df = file["img_url"]
count = 0
for row in df:

    url = row
    print(url)
    result = requests.get(url, stream=True)
    if result.status_code == 200:
        image = result.raw.read()
        open(Output+"/image"+str(count) + ".jpg","wb").write(image)
        count += 1