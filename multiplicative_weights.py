import numpy as np
import time
import logging as log

def run_mwu(models, iters, X, Y, noise_budget, noise_func, loss, epsilon=None):

    num_models = len(models)
    num_points = X.size()[0]

    # compute epsilon as a function of the number of rounds, see paper for more details
    if epsilon is None:
        delta = np.sqrt(4 * np.log(num_models) / float(iters))
        epsilon = delta / 2.0
    else:
        delta = 2.0 * epsilon

    log.info("\nRunning MWU for {} Iterations with Epsilon {}\n".format(iters, epsilon))
    log.info("Guaranteed to be within {} of the minimax value \n".format(delta))

    # lets first do it for just one point and then add the rest of the functionality for multiple points
    weights = np.ones((num_points, num_models)) / num_models
    noise_vectors = []

    for t in range(iters):
        log.debug("Iteration {}\n".format(t))
        start_time = time.time()

        # best response is a sequence of m vectors, m=num_points
        noise_vectors_t = []

        for m in range(num_points):

            best_response = adversary(weights, models, X[m], Y[m], noise_budget, noise_func)
            noise_vectors_t.append(best_response)

            # current loss is an array of length num_models
            # TODO
            current_loss = loss(models, best_response[m], X[m], Y[m])

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





        log.debug("Maximum (Average) Accuracy of Classifier {}".format(acc_history[-1]))
        if dl:
            log.debug("Cost (Before Noise) {}".format(np.array([1 - model.evaluate(X, Y, verbose=0)[1] for model in models])))
        else:
            log.debug("Cost (Before Noise) {}".format(np.array([1 - model.evaluate(X, Y) for model in models])))

        log.debug("Cost (After Noise), {}".format(cost_t))
        log.debug("Loss {} Loss Per Action {}".format(loss, individual))







        log.debug("time spent {}\n".format(time.time() - start_time))
    log.info("finished running MWU ")
    return w, v, loss_history, acc_history, action_loss
