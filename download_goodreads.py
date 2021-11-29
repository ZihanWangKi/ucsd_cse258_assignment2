import sys
import pandas as pd
import gdown
import os

DIR = '/data/zihan/coursework/cse258/assignment2'
file_ids = pd.read_csv(os.path.join(DIR, 'goodreads', 'gdrive_id.csv'))
print(file_ids)

file_id_map = dict(zip(file_ids['name'].values, file_ids['id'].values))

def download_by_name(output=None, quiet=False):
    for fname in file_id_map.values():
        url = 'https://drive.google.com/uc?id='+ fname
        print(fname, url)
        gdown.download(url, output=output, quiet=quiet)

os.makedirs(os.path.join(DIR, "data", "goodreads"), exist_ok=True)
os.chdir(os.path.join(DIR, "data", "goodreads"))
download_by_name()
