import torch

from ReviewClassifier import ReviewClassifier
from ReviewDataset import ReviewDataset
from helpers import predict_rating
from settings import args

dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv, args.vectorizer_file)
vectorizer = dataset.get_vectorizer()

classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier.load_state_dict(torch.load("{}".format(args.model_state_file)))

test_review = 'this is a pretty awesome book'

classifier = classifier.cpu()
prediction = predict_rating(
    test_review,
    classifier,
    vectorizer,
    decision_threshold=0.5
)
print("{} => {}".format(test_review, prediction))
