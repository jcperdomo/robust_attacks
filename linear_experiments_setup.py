from torchvision import datasets, transforms
import numpy as np
import argparse
from sklearn.svm import LinearSVC
from torch_models import BinaryClassifier, MultiClassifier
import torch
import logging as log
import os
import sys


def subset_multiclass_data(data, labels, label_dict):
    # subsets data to only include labels in label dict
    # label dict has the form of original label -> new label
    subset = set(label_dict.keys())
    X = []
    Y = []
    for i in range(len(data)):
        label = labels[i]
        if label in subset:
            label = label_dict[label]
            X.append(data[i])
            Y.append(label)
    return np.array(X), np.array(Y)


def get_mnist_data(train, sel_labels):
    """

    :param train: bool, indicates whether to download the training set or test set
    :param sel_labels: list of labels to subset from the full data set
    :return: X, Y
    """
    mnist = datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor())
    if train:
        images = mnist.train_data.numpy().reshape(-1, 28 * 28) / 255.0
        labels = mnist.train_labels.numpy()
    else:
        images = mnist.test_data.numpy().reshape(-1, 28 * 28) / 255.0
        labels = mnist.test_labels.numpy()

    if len(sel_labels) == 2:
        label_dict = {sel_labels[0]: -1, sel_labels[1]: 1}
    else:
        label_dict = {sel_labels[i]: i for i in range(len(sel_labels))}

    return subset_multiclass_data(images, labels, label_dict)


def generate_feature_independent_svms(num_classifiers, train_data, train_labels):
    """

    :param num_classifiers: int
    :param train_data: numpy array
    :param train_labels: numpy array
    :return: list of torch models with weight vectors that are nonzero in mutually exclusive dimensions
    """
    dims = train_data.shape[1]
    dim_per_classifier = int(dims / num_classifiers)  # defaults to floor function if not integer
    remaining_dims = list(range(dims))
    chosen_dims = []
    for _ in range(num_classifiers):
        chosen = np.random.choice(remaining_dims, size=dim_per_classifier, replace=False)
        chosen_dims.append(chosen)
        remaining_dims = list(set(remaining_dims).difference(set(chosen)))

    models = []
    zeroed_features_list = [list(set(range(dims)).difference(set(x))) for x in chosen_dims]

    for i in range(num_classifiers):
        sparse_data = np.copy(train_data)
        sparse_data[:, zeroed_features_list[i]] = 0.0
        model = LinearSVC(loss='hinge')
        model.fit(sparse_data, train_labels)
        models.append(model)

    TorchModel = BinaryClassifier if len(set(train_labels)) == 2 else MultiClassifier

    torch_models = [TorchModel(torch.tensor(model.coef_, dtype=torch.float),
                               torch.tensor(model.intercept_, dtype=torch.float)) for model in models]

    return torch_models

def generate_experiment_data(num_pts, X, Y, models):
    # returns num_pts from (X, Y) that are correctly classified by all models
    num_selected = 0
    num_models = len(models)
    res_X = []
    res_Y = []
    for i in range(len(X)):
        all_correct = sum([model.accuracy(X[i:i + 1], Y[i:i + 1]) for model in models]) == num_models
        if all_correct:
            res_X.append(X[i])
            res_Y.append(Y[i])
            num_selected += 1
        if num_selected == num_pts:
            break
    if num_selected < num_pts:
        log.info("Not enough points were correctly predicted by all models")
    return torch.stack(res_X), torch.stack(res_Y)

def main(arguments):

    parser = argparse.ArgumentParser()
    parser.add_argument("-num_points", help="number of points to generate", type=int, required=True)
    parser.add_argument("-num_classifiers", help="number of classifiers", type=int, required=True)
    parser.add_argument("-sel_labels", help="list of labels to subset", nargs='+', type=int, required=True)
    args = parser.parse_args(arguments)

    subdirectory = 'binary' if len(args.sel_labels) == 2 else 'multi'

    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='experiment_data/linear_{}_setup.log'.format(subdirectory), filemode='w')
    log.info('Number of Points {}'.format(args.num_points))
    log.info('Number of Classifiers {}'.format(args.num_classifiers))
    log.info('Selected Labels {}'.format(args.sel_labels))

    train_data, train_labels = get_mnist_data(True, args.sel_labels)
    test_data, test_labels = get_mnist_data(False, args.sel_labels)

    models = generate_feature_independent_svms(args.num_classifiers, train_data, train_labels)

    model_save_path = 'models/' + subdirectory + '/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    test_data = torch.tensor(test_data, dtype=torch.float)
    test_labels = torch.tensor(test_labels)
    exp_images, exp_labels = generate_experiment_data(args.num_points, test_data, test_labels, models)

    distances = np.array([model.distance(exp_images).detach().numpy() for model in models])
    percentiles = [np.percentile(dist, [10, 25, 50, 75, 90]) for dist in distances]
    log.info("Distance Percentiles, Table [models] x percentiles [10, 25, 50, 75, 90]\n")
    for i, p in enumerate(percentiles):
        log.info("Model {} Distance Percentiles {}".format(i, list(p)))
    log.info("\n")

    for i, model in enumerate(models):
        log.info('Model {} Test Accuracy : {}'.format(i, model.accuracy(test_data, test_labels)))
        torch.save(model, model_save_path + 'model_{}.pt'.format(i))

    data_save_path = 'experiment_data/linear/' + subdirectory + '/'
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)

    torch.save(exp_images, 'experiment_data/linear/' + subdirectory + '/mnist_images.pt')
    torch.save(exp_labels, 'experiment_data/linear/' + subdirectory + '/mnist_labels.pt')

    log.info('Success')


if __name__ == "__main__":
    main(sys.argv[1:])
