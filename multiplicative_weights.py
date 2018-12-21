import numpy as np
import time
import torch


def run_mwu(models, iters, X, Y, noise_budget, adversary, epsilon=None):
    num_models = len(models)
    num_points = X.size()[0]

    # compute epsilon as a function of the number of rounds, see paper for more details
    if epsilon is None:
        delta = np.sqrt(4 * np.log(num_models) / float(iters))
        epsilon = delta / 2.0
    else:
        delta = 2.0 * epsilon

    print("\nRunning MWU for {} Iterations with Epsilon {}\n".format(iters, epsilon))
    print("Guaranteed to be within {} of the minimax value \n".format(delta))

    # lets first do it for just one point and then add the rest of the functionality for multiple points
    weights = np.ones((num_points, num_models)) / num_models
    expected_losses = [[] for _ in range(num_points)]
    minimum_losses = [[] for _ in range(num_points)]
    noise_vectors = []

    for t in range(iters):
        print("Iteration {}\n".format(t))
        start_time = time.time()

        # best response is a sequence of m vectors, m=num_points
        noise_vectors_t = []

        for m in range(num_points):

            x = X[m].unsqueeze(0).cuda()
            y = Y[m].cuda()

            best_response = adversary(weights[m], models, x, y, noise_budget)

            # current loss is an array of length num_models
            # TODO
            current_loss = np.array([1.0 - model.loss_single(x + best_response, y).item() for model in models])
            expected_losses[m].append(np.dot(weights[m], current_loss))
            minimum_losses[m].append(current_loss.min())

            noise_vectors_t.append(best_response.cpu())

            # penalize experts
            for i in range(num_models):
                weights[m, i] = weights[m, i] * (1.0 - epsilon) ** current_loss[i]

            # renormalize weights
            weights_sum = weights[m].sum()
            for i in range(num_models - 1):
                weights[m, i] = weights[m, i] / weights_sum
            weights[m, -1] = 1.0 - weights[m, :-1].sum()

        noise_vectors_t = torch.stack(noise_vectors_t)
        noise_vectors.append(noise_vectors_t)

        print("time spent {}\n".format(time.time() - start_time))
    noise_vectors = torch.stack(noise_vectors)
    print("finished running MWU ")
    return noise_vectors, weights, expected_losses, minimum_losses
