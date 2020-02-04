import torch
import torch.optim as optim
from torch import nn
from tqdm import notebook

from ReviewClassifier import ReviewClassifier
from ReviewDataset import ReviewDataset
from generate_batches import generate_batches
from helpers import make_train_state, compute_accuracy, update_train_state, predict_rating
from settings import args

if args.reload_from_files:
    print('Loading dataset and vectorizer')
    dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv, args.vectorizer_file)
else:
    print('Loading dataset and creating vectorizer')
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()

classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

train_state = make_train_state(args)
epoch_bar = notebook.tqdm(desc='training routine', total=args.num_epoch, position=0)

dataset.set_split('train')
train_bar = notebook.tqdm(
    desc='split=train',
    total=dataset.get_num_batches(args.batch_size),
    position=1,
    leave=True
)

dataset.set_split('val')
val_bar = notebook.tqdm(
    desc='split=val',
    total=dataset.get_num_batches(args.batch_size),
    position=1,
    leave=True
)

try:
    for epoch_index in range(args.num_epoch):
        train_state['epoch_index'] = epoch_index

        dataset.set_split('train')
        batch_generator = generate_batches(
            dataset,
            batch_size=args.batch_size,
            device=args.device
        )
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            print("epoch_index: {} => batch_index: {}".format(epoch_index, batch_index))
            # 1. zero the gradients
            optimizer.zero_grad()

            # 2. compute the output
            y_pred = classifier(x_in=batch_dict['x_data'].float())

            # 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # 4. use loss to produce gradients
            loss.backward()

            # 5. use optimizer to take gradient step
            optimizer.step()

            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_bar.set_postfix(
                loss=running_loss,
                acc=running_acc,
                epoch=epoch_index
            )
            train_bar.update()

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        train_state = update_train_state(args=args, model=classifier, train_state=train_state)

        scheduler.step(train_state['val_loss'][-1])

        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()

        if train_state['stop_early']:
            break

        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()
except KeyboardInterrupt:
    print('Exiting loop')


# compute the loss & accuracy on the test set using the best available model
classifier.load_state_dict(torch.load(train_state['model_filename']))
classifier = classifier.to(args.device)

dataset.set_split('test')
batch_generator = generate_batches(
    dataset,
    batch_size=args.batch_size,
    device=args.device
)

running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # 1. compute the output
    y_pred = classifier(x_in=batch_dict['x_data'].float())

    # 2. compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'].float())
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    # 3. compute the accuracy
    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

print("Test loss: {:.3f}".format(train_state['test_loss']))
print("Test Accuracy: {:.2f}".format(train_state['test_acc']))


test_review = 'this is a pretty awesome book'

classifier = classifier.cpu()
prediction = predict_rating(
    test_review,
    classifier,
    vectorizer,
    decision_threshold=0.5
)
print("{} => {}".format(test_review, prediction))
