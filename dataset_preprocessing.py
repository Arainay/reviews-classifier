import collections
import re
import numpy as np
import pandas as pd

from argparse import Namespace

args = Namespace(
    raw_train_dataset='data/raw_train.csv',
    raw_test_dataset='data/raw_test.csv',
    ouput_dataset='data/reviews.csv',
    proportion_subset_of_train=0.1,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    seed=1337
)

train_reviews = pd.read_csv(
    args.raw_train_dataset,
    header=None,
    names=['rating', 'review']
)

print(train_reviews.info())
print()
print(train_reviews.describe())
print()
print(train_reviews['rating'].value_counts())

by_rating = collections.defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row['rating']].append(row.to_dict())

reviews = []
np.random.seed(args.seed)

for _, item_list in sorted(by_rating.items()):

    np.random.shuffle(item_list)

    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)
    n_test = int(args.test_proportion * n_total)

    for item in item_list[:n_train]:
        item['split'] = 'train'

    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'

    for item in item_list[n_train+n_val:n_train+n_val+n_test]:
        item['split'] = 'test'

    reviews.extend(item_list)

reviews = pd.DataFrame(reviews)

print(reviews['split'].value_counts())


def handle_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


reviews['review'] = reviews['review'].apply(handle_text)
reviews['rating'] = reviews['rating'].apply({1: 'negative', 2: 'positive'}.get)

print(reviews.head())

reviews.to_csv(args.ouput_dataset)
