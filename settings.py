import torch
import os
from argparse import Namespace

from general_utilities import set_seed_everywhere, handle_dirs

args = Namespace(
    frequency_cutoff=25,
    model_state_file='model.pth',
    review_csv='data/reviews.csv',
    save_dir='model_storage',
    vectorizer_file='vectorizer.json',
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epoch=100,
    seed=1337,
    catch_keyboard_inteppupt=True,
    cuda=False,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False
)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
    args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))

if torch.cuda.is_available():
    args.cuda = True

print("Using CUDA: {}".format(args.cuda))

args.device = torch.device("cuda" if args.cuda else "cpu")

set_seed_everywhere(args.seed, args.cuda)

handle_dirs(args.save_dir)
