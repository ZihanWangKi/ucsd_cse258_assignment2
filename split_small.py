import json
import os.path
import random
import sys

dataset = sys.argv[1]
assert dataset in ["amazon", "goodreads"]

path = f"/data/zihan/coursework/cse258/assignment2/data/{dataset}"

with open(os.path.join(path, "train.json")) as f:
    data = [json.loads(x) for x in f.readlines()]

def save_json(data, path_to_save):
    with open(path_to_save, "w") as f:
        for review_entry in data:
            f.write(json.dumps(review_entry))
            f.write("\n")

save_json(data[: 500], os.path.join(path, "train_500.json"))
save_json(data[: 200], os.path.join(path, "train_200.json"))
save_json(data[: 100], os.path.join(path, "train_100.json"))
