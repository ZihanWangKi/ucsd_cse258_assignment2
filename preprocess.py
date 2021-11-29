import json
import os.path
import random
import sys

dataset = sys.argv[1]
assert dataset in ["amazon", "goodreads"]

path = f"/data/zihan/coursework/cse258/assignment2/data/{dataset}"
json_name = {
    "amazon": "reviews_Books_5.json",
    "goodreads": "goodreads_reviews_dedup.json"
}
review_text_name = {
    "amazon": "reviewText",
    "goodreads": "review_text"
}
review_rating_name = {
    "amazon": "overall",
    "goodreads": "rating"
}
path_to_data = os.path.join(path, json_name[dataset])

review_data = []
with open(path_to_data, "r") as f:
    line = f.readline()
    while line:
        review_entry = json.loads(line)
        rating = review_entry[review_rating_name[dataset]]
        rating_integer = round(rating)
        assert abs(rating_integer - rating) < 1e-6
        if rating_integer == 0:
            line = f.readline()
            continue
        assert 1 <= rating_integer <= 5
        review_data.append({"review_text": review_entry[review_text_name[dataset]], "label": rating_integer})
        line = f.readline()
        if len(review_data) % 10000 == 0:
            print(f"{len(review_data)} Done.")
        if len(review_data) == 100000:
            break

def save_json(data, path_to_save):
    with open(path_to_save, "w") as f:
        for review_entry in data:
            f.write(json.dumps(review_entry))
            f.write("\n")

save_json(review_data, os.path.join(path, "all.json"))

FULL_TRAIN_SIZE = 10000
FULL_EVAL_SIZE = 10000

random.seed(42)
random.shuffle(review_data)
save_json(review_data[: FULL_TRAIN_SIZE], os.path.join(path, "train.json"))
save_json(review_data[FULL_TRAIN_SIZE: FULL_TRAIN_SIZE + FULL_EVAL_SIZE], os.path.join(path, "eval.json"))
