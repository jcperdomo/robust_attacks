import numpy as np
from attacks import try_region_binary, distributional_oracle_multi, distributional_oracle_binary
from torch_models import BinaryClassifier, MultiClassifier, try_region_multi
import torch
import sys

def subset_feasible_models(models, x, noise_budget):
    dists = [model.distance(x).item() for model in models]
    num_models = len(models)
    return [models[i] for i in range(num_models) if dists[i] < noise_budget]


def ensemble_linear_models(models):
    num_classifiers = len(models)
    model_type = type(models[0])
    if model_type is BinaryClassifier:
        TorchModel = BinaryClassifier
        ensemble_weights = sum([1.0 / num_classifiers * model.weights.reshape(1, -1)
                                for model in models])

    else:
        TorchModel = MultiClassifier
        ensemble_weights = sum([1.0 / num_classifiers * model.weights
                                for model in models])

    ensemble_bias = sum([1.0 / num_classifiers * model.bias for model in models])
    ensemble_weights = torch.tensor(ensemble_weights, dtype=torch.float)
    ensemble_bias = torch.tensor(ensemble_bias, dtype=torch.float)
    ensemble = TorchModel(ensemble_weights, ensemble_bias)
    return ensemble


def compute_linear_ensemble_baseline(models, images, labels, noise_budget):
    model_type = type(models[0])
    if model_type is BinaryClassifier:
        oracle = distributional_oracle_binary
        out_dim = 1
    else:
        oracle = distributional_oracle_multi
        out_dim = 3

    noise_vectors = []
    for i in range(len(images)):
        x = images[i]  # .unsqueeze()
        y = labels[i]
        ensemble = ensemble_linear_models(subset_feasible_models(models, x, noise_budget))
        ensemble_array = [(torch.tensor(ensemble.weights.reshape(out_dim, -1),
                                        dtype=torch.float),
                           torch.tensor(ensemble.bias, dtype=torch.float))]

        v = oracle(np.ones(1), ensemble_array, x, y, sys.maxsize)
        v = v / v.norm() * noise_budget
        noise_vectors.append(v)
    return torch.stack(noise_vectors).reshape(images.size())


def coordinate_ascent(models, x, y, noise_budget):
    models = subset_feasible_models(models, x, noise_budget)
    num_models = len(models)

    sol = torch.zeros(x.size())
    # can't trick anything
    if num_models == 0:
        return torch.zeros(x.size())

    model_type = type(models[0])
    if model_type is BinaryClassifier:
        try_region = try_region_binary
        labels = [-1, 1]
    else:
        try_region = try_region_multi
        labels = range(3)

    x = x.numpy().reshape(-1,)
    y = y.item()

    label_vector = [y] * num_models  # initialize to the original point, of length feasible_models
    label_options = list(set(labels).difference(set([y])))
    model_options = list(range(num_models))

    for i in range(num_models):
        coord = np.random.choice(model_options)
        model_options = list(set(model_options).difference([coord]))

        label_vector[coord] = np.random.choice(label_options)
        v = try_region(models, label_vector, x)

        if v is not None:
            norm = np.linalg.norm(v)
            if norm <= noise_budget:
                sol = torch.tensor(v, dtype=torch.float32).reshape(1,-1)
            else:
                break
        else:
            break

    return sol


def compute_linear_coordinate_ascent_baseline(models, images, labels, noise_budget):
    coordinate_ascent_baseline = []
    for i in range(len(images)):
        x = images[i]
        y = labels[i]
        coordinate_ascent_baseline.append(coordinate_ascent(models, x, y, noise_budget))
    return torch.stack(coordinate_ascent_baseline).reshape(images.size())


def compute_max_individual_baseline(models, images, labels, noise_budget):
    model_type = type(models[0])
    if model_type is BinaryClassifier:
        oracle = distributional_oracle_binary
        out_dim = 1
    else:
        oracle = distributional_oracle_multi
        out_dim = 3

    noise_vectors = []
    for i in range(len(images)):
        x = images[i]
        y = labels[i]

        individual_attacks = []
        for model in models:
            model_array = [(torch.tensor(model.weights.reshape(out_dim, -1),
                                         dtype=torch.float),
                            torch.tensor(model.bias, dtype=torch.float))]

            v = oracle(np.ones(1), model_array, x, y, sys.maxsize)
            v = v / v.norm() * noise_budget

            max_acc = max([model.accuracy(x + v, y) for model in models])
            individual_attacks.append((max_acc, v))

        noise_vectors.append(min(individual_attacks, key=lambda x: x[0])[1])

    return torch.stack(noise_vectors).reshape(images.size())


