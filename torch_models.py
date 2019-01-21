import torch.nn as nn
import torch
import torchvision
import numpy as np
import os
from cvxopt import matrix, solvers

# from attacks import try_region_mult

def try_region_multi(models, labels, x, delta=1e-5, num_labels=3):
    dim = x.shape[0]
    P = matrix(np.identity(dim))
    q = matrix(np.zeros(dim))
    h = []
    G = []
    num_models = len(models)
    for i in range(num_models):
        others = list(range(num_labels))
        target = labels[i]
        del others[target]
        target_w, target_b = models[i].weights[target], models[i].bias[target]
        for j in others:
            other_w, other_b = models[i].weights[j], models[i].bias[j]
            ineq_val = np.dot(target_w - other_w, x) + target_b - other_b - delta
            h.append(ineq_val)
            G.append(other_w - target_w)
    G = np.concatenate([np.array(G), -1 * np.identity(dim), np.identity(dim)])
    h = matrix(np.concatenate([h, x, 1.0 - x]))  # assumes inputs need to lie in [0,1]
    G = matrix(np.array(G, dtype=np.float64))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    if sol['status'] == 'optimal':
        v = np.array(sol['x']).reshape(-1, )
        perturbed_x = torch.tensor((x + v).reshape(1, -1), dtype=torch.float)
        is_desired_label = [models[i].predict(perturbed_x).item() == labels[i] for i in range(num_models)]
        if sum(is_desired_label) == num_models:
            return v
        else:
            print('looped')
            return try_region_multi(models, labels, x, delta * 1.5, num_labels)
    else:
        return None


class BinaryClassifier(nn.Module):

    def __init__(self, weights, bias):
        super(BinaryClassifier, self).__init__()

        # normalize weights to make distance measurements more convenient
        norm = weights.norm(2)
        weights = weights / norm
        bias = bias / norm

        in_dim, out_dim = weights.size()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight = nn.Parameter(weights)
        self.linear.bias = nn.Parameter(bias)

        # maintain numpy versions for (ease of) use with convex best response oracle
        self.weights = weights.numpy().reshape(-1, )
        self.bias = bias.item()

        self.oracle = False

    def forward(self, x):
        return self.linear(x).reshape(-1, )

    def predict(self, x):
        return torch.sign(self.forward(x))

    def distance(self, x):
        # returns distance of points to the decision boundary
        return torch.abs(self.forward(x))

    def accuracy(self, X, Y):
        Y = Y.float()
        return (torch.sign(self.forward(X)) == Y).to(torch.float).mean().item()

    def loss_single(self, x, v, y, noise_budget):
        if self.oracle:
            return torch.tensor(self.accuracy(x + v, y))
        else:
            # implements the reverse hinge loss, normalized to be bounded in the range [0,1]
            relu = nn.ReLU()
            """
            noise_budget * -y / ||w|| * w  is the optimal attack
            therefore, max loss is induced by pushing in the opposite direction
            """
            y = y.float()
            max_loss = (y * self.forward(x + y * noise_budget * self.linear.weight.data)).item()
            # it's possible that the max loss is less than 0 if the point is very far on the incorrect
            # side of the decision boundary
            max_loss = 1 if max_loss < 0 else max_loss
            return relu(y * self.forward(x + v)) / max_loss


class MultiClassifier(nn.Module):

    def __init__(self, weights, bias):
        super(MultiClassifier, self).__init__()

        out_dim, in_dim = weights.size()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight = nn.Parameter(weights)
        self.linear.bias = nn.Parameter(bias)

        # maintain numpy versions for use with convex best response oracle
        self.weights = weights.numpy()
        self.bias = bias.numpy()
        self.oracle = False

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        out = self.forward(x)
        _, preds = torch.max(out, 1)
        return preds

    def distance(self, X):
        """
        Computes the minimum distance from a point to the decision boundary
        by finding the optimal perturbation for each targeted attack and choosing the minium

        returns: a vector of shape (num_points,) with the corresponding distances
        """
        n = X.shape[0]
        Y = self.predict(X)
        X = X.numpy()

        distances = []
        for i in range(n):
            label_options = list(range(self.weights.shape[0]))  # create list of length num_classes
            del label_options[Y[i]]
            dists = []
            for j in label_options:
                v = try_region_multi([self], [j], X[i])
                dists.append(np.linalg.norm(v))
            distances.append(min(dists))
        return torch.tensor(distances)

    def accuracy(self, X, Y):
        out = self.forward(X)
        _, preds = torch.max(out, 1)
        return (preds == Y).to(torch.float).mean().item()

    def loss_single(self, x, v, y, noise_budget):
        if self.oracle:
            return torch.tensor(self.accuracy(x + v, y))
        else:
            logits = self.forward(x + v)
            # probs
            # true_max, max_ix = torch.max(probs, 1)
            logits2 = logits.clone()
            logits2[0, y.item()] = torch.min(logits) - 1.0
            _, second_max_ix = torch.max(logits2, 1)
            diff = logits[0, y.item()] - logits[0, second_max_ix.item()]
            relu = nn.ReLU()
            sigmoid = nn.Sigmoid()
            loss = (sigmoid(diff) - .5) * 2
            return relu(loss)


class DNN(nn.Module):
    
    def __init__(self, model, cuda=True):
        super(DNN, self).__init__()
        # self.softmax = nn.Softmax(dim=1)
        self.model = model
#         if cuda:
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.var = torch.tensor([0.229, 0.224, 0.225]).pow(2).cuda()
#         else:
#             self.mean = torch.tensor([0.485, 0.456, 0.406])
#             self.var = torch.tensor([0.229, 0.224, 0.225]).pow(2)

    def forward(self, x):
        x = nn.functional.batch_norm(x, self.mean, self.var, momentum=0.0, eps=0.0)
        logits = self.model(x)
        # return self.softmax(logits)
        return logits

    def loss(self, X, Y):
        res = torch.zeros(X.size()[0])
        probs = self.forward(X)
        true_max, preds = torch.max(probs, 1)
        correct = (preds == Y).to(torch.uint8).cpu()
        correct_ixs = torch.masked_select(torch.tensor(range(X.size()[0]), dtype=torch.long),
                                          correct)
        true_max = true_max[correct_ixs].clone().cpu()
        probs[:, preds] = -1.0
        second_max, _ = torch.max(probs, 1)
        second_max = second_max[correct_ixs].cpu()
        res[correct_ixs] = true_max - second_max
        return res

    def accuracy(self, X, Y, batch=False):
        if batch:
            accs = []
            for x, y in zip(X, Y):
                out = self.forward(x.unsqueeze(0))
                _, preds = torch.max(out, 1)
                accs.append((preds == y.unsqueeze(0)).to(torch.float).item())
            res = np.mean(accs)
        else:
            out = self.forward(X)
            _, preds = torch.max(out, 1)
            res = (preds == Y).to(torch.float).mean().item()
        return res

    def loss_single(self, x, v, y, noise_budget):
        # probs = self.forward(x + v)
        # # true_max, max_ix = torch.max(probs, 1)
        # probs2 = probs.clone()
        # probs2[0, y.item()] = -1.0
        # _, second_max_ix = torch.max(probs2, 1)
        # relu = nn.ReLU()
        # return relu(probs[0, y.item()] - probs[0, second_max_ix.item()])
        logits = self.forward(x + v)
        logits2 = logits.clone()
        logits2[0, y.item()] = torch.min(logits) - 1.0
        _, second_max_ix = torch.max(logits2, 1)
        diff = logits[0, y.item()] - logits[0, second_max_ix.item()]
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        loss = (sigmoid(diff) - .5) * 2
        return relu(loss)


def load_models(exp_type):

    if exp_type == 'imagenet':
        resnet18 = torchvision.models.resnet18(pretrained=True).cuda().eval()
        resnet50 = torchvision.models.resnet50(pretrained=True).cuda().eval()
        vgg13 = torchvision.models.vgg13(pretrained=True).cuda().eval()
        vgg19 = torchvision.models.vgg19_bn(pretrained=True).cuda().eval()
        densenet = torchvision.models.densenet161(pretrained=True).cuda().eval()
        models = [DNN(resnet18), DNN(resnet50), DNN(vgg13), DNN(vgg19), DNN(densenet)]

    elif exp_type == "mnist_binary":
        models = [torch.load('models/binary/' + model_file) for model_file in os.listdir('models/binary/')]

    elif exp_type == 'mnist_multi':
        models = [torch.load('models/multi/' + model_file) for model_file in os.listdir('models/multi/')]

    return models





