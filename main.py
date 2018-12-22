import logging as log
import argparse
from multiplicative_weights import run_mwu
import sys
from functools import partial
from torch_models import load_imagenet_models
import os
import attacks
import time
import torch
import numpy as np


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_type", help="choice of experiment to run",
                        choices=["imagenet", "mnist_binary", "mnist_multiclass"], required=True)
    parser.add_argument("-mwu_iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-noise_budget", help="noise budget", type=float, required=True)
    parser.add_argument("-pgd_iters", help='number of iterations to run (projected gradient descent)', type=int)
    parser.add_argument("-name", help='name (used to name results directory)', type=str, required=True)
    args = parser.parse_args(arguments)

    if not os.path.exists('experiment_results/'):
        os.mkdir('experiment_results/')

    exp_dir = 'experiment_results/' + args.name
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename=exp_dir + "/results.log", filemode='w')
    log.info("Experiment Type {}".format(args.exp_type))
    log.info("MWU Iters {} ".format(args.mwu_iters))
    if args.pgd_iters:
        log.info("PGD Iters {}".format(args.pgd_iters))
    log.info("Noise Budget {}".format(args.noise_budget))

    X_exp = torch.load('experiment_data/imagenet_images.pt')[:3]
    Y_exp  = torch.load('experiment_data/imagenet_labels.pt')[:3]

    log.info("Num Points {}".format(X_exp.size()[0]))

    if args.exp_type == 'imagenet':
        models = load_imagenet_models()
        adversary = partial(attacks.pgd, iters=args.pgd_iters)


    noise_vectors, weights, expected_losses, minimum_losses = run_mwu(models, args.mwu_iters, X_exp, Y_exp,
                                                                      args.noise_budget, adversary)
    torch.save(noise_vectors, exp_dir + 'noise_vectors.pt')

    np.save(exp_dir  + "/weights.npy", weights)
    np.save(exp_dir  + "/expected_losses.npy", expected_losses)
    np.save(exp_dir  + "/minimum_losses.npy", minimum_losses)

    log.info("Finished!")

if __name__ == "__main__":
    main(sys.argv[1:])
