import numpy as np
import time
import torch
import logging as log
# import ray
import multiprocessing

def run_mwu(models, iters, X, Y, noise_budget, adversary, cuda, use_ray, epsilon=None):
    num_models = len(models)
    num_points = X.size()[0]

    # compute epsilon as a function of the number of rounds, see paper for more details
    if epsilon is None:
        delta = np.sqrt(4 * np.log(num_models) / float(iters))
        epsilon = min(delta / 2.0, .99)
    else:
        delta = 2.0 * epsilon

    log.info("\nRunning MWU for {} Iterations with Epsilon {}\n".format(iters, epsilon))
    log.info("Guaranteed to be within {} of the minimax value \n".format(delta))

    # lets first do it for just one point and then add the rest of the functionality for multiple points
    weights = np.ones((num_points, num_models)) / num_models
    expected_losses = [[] for _ in range(num_points)]
    minimum_losses = [[] for _ in range(num_points)]
    noise_vectors = []

    #TODO
    if use_ray:
        model_arrays = [(torch.tensor(model.weights.reshape(1,-1), dtype=torch.float), torch.tensor(model.bias, dtype=torch.float))
                        for model in models]

    for t in range(iters):
        log.info("Iteration {}\n".format(t))
        start_time = time.time()

        # best response is a sequence of m vectors, m=num_points
        noise_vectors_t = []

        # TODO parallelize the oracle
        if use_ray:
            print('using multiprocessing')
            param_list = []
            for m in range(num_points):
                x = X[m].unsqueeze(0)
                y = Y[m] #TODO
                param_list.append((weights[m], model_arrays, x, y, noise_budget))
            with multiprocessing.Pool(processes=30) as pool:
                best_responses = pool.starmap(adversary, param_list)

        for m in range(num_points):

            x = X[m].unsqueeze(0)
            y = Y[m]

            if cuda:
                x = x.cuda()
                y = y.cuda()

            if use_ray:
                best_response = best_responses[m]
            else:
                # calculate the adversary's response given current distribution
                best_response = adversary(weights[m], models, x, y, noise_budget)

            # compute loss of learner per expert
            current_loss = np.array([1.0 - model.loss_single(x, best_response, y, noise_budget).item() for model in models])

            expected_losses[m].append(np.dot(weights[m], current_loss))
            minimum_losses[m].append(current_loss.min())

            noise_vectors_t.append(best_response.cpu())

            # penalize experts
            for i in range(num_models):
                weights[m, i] = weights[m, i] * (1.0 - epsilon) ** current_loss[i]

            # normalize weights
            weights_sum = weights[m].sum()
            for i in range(num_models - 1):
                weights[m, i] = weights[m, i] / weights_sum
            weights[m, -1] = 1.0 - weights[m, :-1].sum()

        noise_vectors_t = torch.stack(noise_vectors_t)
        noise_vectors.append(noise_vectors_t)

        time_spent = time.time() - start_time
        log.info("time spent {}".format(time_spent))
        log.info("time spent per point {}\n".format(time_spent / num_points))

    noise_vectors = torch.stack(noise_vectors)
    log.info("finished running multiplicative weights ")
    return noise_vectors, weights, np.array(expected_losses), np.array(minimum_losses)
