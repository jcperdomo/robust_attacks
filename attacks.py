import torch
from cvxopt import matrix, solvers
from itertools import product
import numpy as np
import ray

def pgd(weights, models, x, y, noise_budget, iters, clip_min=0.0, clip_max=1.0, cuda=True):
    step_size = noise_budget / (.8 * iters)
    
    # initialize the vector we will be optimizing
    curr_noise_vector = torch.zeros(x.size())
    if cuda:
        curr_noise_vector= curr_noise_vector.cuda()

    for i in range(iters):
        # variables we will be using this iteration
        total_loss = torch.zeros(1)
        var_noise_vector = torch.autograd.Variable(curr_noise_vector, requires_grad=True)
        if cuda:
            var_noise_vector = var_noise_vector.cuda()
            total_loss = total_loss.cuda()
        
        if var_noise_vector.grad is not None:
                var_noise_vector.grad.data.zero_()
        
        # compute weighted sum of losses
        for w, model in zip(weights, models):
            loss = w * model.loss_single(x, var_noise_vector, y, noise_budget)
            total_loss += loss

        total_loss.backward()
        grad = var_noise_vector.grad.data
        # update variables respecting norm and box contraints
        grad_norm = grad.norm(2)
        if grad_norm > 0:

            curr_noise_vector += -1 * step_size * grad / grad_norm
            noise_norm = torch.norm(curr_noise_vector, p=2)

            if  noise_norm > noise_budget:
                curr_noise_vector = noise_budget * curr_noise_vector / noise_norm
        
            curr_noise_vector = torch.clamp(x + curr_noise_vector, min=clip_min, max=clip_max) - x
          
        else:
            break
    return curr_noise_vector


def try_region_binary(models, signs, x, delta=1e-5):
    """
    models: list of LinearBinaryClassifiers
    signs: list of signs (+1, -1) of length num_models
    x: np array of shape dim (a single point)
    returns: a vector in the region denoted by the signs vector
    """
    dim = x.shape[0]
    P = matrix(np.identity(dim))
    q = matrix(np.zeros(dim))
    h = []
    G = []
    num_models = len(models)
    for i in range(num_models):
        weights, bias = models[i].weights, models[i].bias
        ineq_val = -1.0 * delta + signs[i] * (np.dot(weights, x) + bias)
        h.append(ineq_val)
        G.append(-1.0 * signs[i] * weights)
    G = np.concatenate([np.array(G), -1 * np.identity(dim), np.identity(dim)])
    h = matrix(np.concatenate([h, x, 1.0 - x]))  # assumes inputs need to lie in [0,1]
    G = matrix(np.array(G, dtype=np.float64))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    if sol['status'] == 'optimal':
        v = np.array(sol['x']).reshape(-1, )
        perturbed_x = torch.tensor((x + v).reshape(1, -1), dtype=torch.float)
        is_desired_sign = [np.sign(models[i](perturbed_x).item()) == signs[i] for i in range(num_models)]
        if sum(is_desired_sign) == num_models:
            return v
        else:
            print('looping')
            return try_region_binary(models, signs, x, delta * 1.5)
    else:
        return None

# TODO parallelize the oracle

# @ray.remote
def distributional_oracle_binary(distribution, models, x, y, noise_budget):
    """
    computes the optimal perturbation for the point (x,y) using convex optimization
    """
    num_models = len(models)
    # we should only take into consideration models that we could feasibly trick
    distances = [model.distance(x).item() for model in models]
    feasible_models = [models[i] for i in range(num_models) if distances[i] < noise_budget]
    distribution = np.array([distribution[i] for i in range(num_models) if distances[i] < noise_budget])
    num_models = len(feasible_models)

    x = x.numpy().reshape(-1, )
    y = y.item()

    # can't trick anything
    if num_models == 0:
        return torch.zeros(x.shape).reshape(1, -1)

    signs_values = []
    for signs in product([-1.0, 1.0], repeat=num_models):  # iterate over all possible regions
        is_misclassified = np.equal(-1.0 * y * np.ones(num_models), signs)  # y = -1, or 1
        value = np.dot(is_misclassified, distribution)
        signs_values.append((signs, value))

    values = sorted(set([value for signs, value in signs_values]), reverse=True)
    for value in values:
        feasible_candidates = []
        for signs in [sign for sign, val in signs_values if val == value]:
            v = try_region_binary(feasible_models, signs, x)
            if v is not None:
                norm = np.linalg.norm(v)
                if norm <= noise_budget:
                    feasible_candidates.append((v, norm))
        # amongst those with the max value, return the one with the minimum norm
        if feasible_candidates:
            # break out of the loop since we have already found the optimal answer
            v = min(feasible_candidates, key=lambda x: x[1])[0]
            return torch.tensor(v, dtype=torch.float32).reshape(1,-1)


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

# @ray.remote
def distributional_oracle_multi(distribution, models, x, y, noise_budget, num_labels=3):
    """
    computes the optimal perturbation for x under alpha and the given distribution
    """
    num_models = len(models)
    # we should only take into consideration models that we could feasibly trick
    distances = [model.distance(x).item() for model in models]
    models = [models[i] for i in range(num_models) if distances[i] < noise_budget]
    distribution = np.array([distribution[i] for i in range(num_models) if distances[i] < noise_budget])
    num_models = len(models)

    x = x.numpy().reshape(-1, )
    y = y.item()

    # can't trick anything
    if num_models == 0:
        return torch.zeros(x.shape).reshape(1, -1)

    num_models = len(models)
    labels_values = []
    for labels in product(range(num_labels), repeat=num_models):  # iterate over all possible regions
        is_misclassified = (np.array(labels) != y).astype(np.float32)
        value = np.dot(is_misclassified, distribution)
        labels_values.append((labels, value))

    values = sorted(set([value for label, value in labels_values]), reverse=True)

    for curr_value in values:
        feasible_candidates = []
        for labels in [labels for labels, val in labels_values if val == curr_value]:
            v = try_region_multi(models, labels, x, num_labels=num_labels)
            if v is not None:
                norm = np.linalg.norm(v)
                if norm <= noise_budget:
                    feasible_candidates.append((v, norm))
        # amongst those with the max value, return the one with the minimum norm
        if feasible_candidates:
            # break out of the loop since we have already found the optimal answer
            v = min(feasible_candidates, key=lambda x: x[1])[0]
            return torch.tensor(v, dtype=torch.float32).reshape(1, -1)

# import torch
## V1, NEED TO TEST V2
# def pgd(weights, models, x, y, noise_budget, iters, clip_min=0.0, clip_max=1.0, cuda=True):
#     step_size = noise_budget / (.8 * iters)
#     noise_vector = torch.zeros(x.size())
#     if cuda: # need to test with the deep learning implementation
#         noise_vector = noise_vector.cuda()
#     #loss_list = []
#     curr_x = x
#     for i in range(iters):
#         var_x = torch.autograd.Variable(curr_x, requires_grad=True).cuda()
#         grad = torch.zeros(x.size()).cuda()
#         if cuda:

#         #total_loss = 0
#         for w, model in zip(weights, models):

#             if var_x.grad is not None:
#                 var_x.grad.data.zero_()

#             loss = w * model.loss_single(var_x, y) / max_loss
#             #total_loss += loss

#             loss.backward()

#             grad += var_x.grad.data
#         #loss_list.append(total_loss)

#         grad_norm = grad.norm(2)
#         if grad_norm > 0:
#             noise_vector += -1 * step_size * grad / grad.norm()
#             noise_norm = torch.norm(noise_vector, p=2)

#             if  noise_norm > noise_budget:
#                 noise_vector = noise_budget * noise_vector / noise_norm

#             curr_x = torch.clamp(x + noise_vector, min=clip_min, max=clip_max)
#         else:
#             break
#     return (curr_x - x)[0] #, loss_list