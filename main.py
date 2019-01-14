import logging as log
import argparse
from multiplicative_weights import run_mwu
import sys
from functools import partial
from torch_models import load_models
import os
import attacks
import time
import torch
import numpy as np


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_type", help="choice of experiment to run",
                        choices=["imagenet", "mnist_binary", "mnist_multi"], required=True)
    parser.add_argument("-mwu_iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument('-noise_function', help='noise function for best response oracle',
                        choices=['oracle', 'pgd'], required=True)
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
    log.info('Noise Function {}'.format(args.noise_function))
    log.info("MWU Iters {} ".format(args.mwu_iters))

    if args.pgd_iters:
        log.info("PGD Iters {}".format(args.pgd_iters))

    log.info("Noise Budget {}".format(args.noise_budget))

    # load data, setup attack
    if args.exp_type == 'imagenet':
        exp_data = torch.load('experiment_data/imagenet_images.pt')[:100]
        exp_labels = torch.load('experiment_data/imagenet_labels.pt')[:100]
        cuda = True

    elif args.exp_type == 'mnist_binary':
        exp_data = torch.load('experiment_data/linear/binary/mnist_images.pt')
        exp_labels = torch.load('experiment_data/linear/binary/mnist_labels.pt')
        cuda = False

    elif args.exp_type == 'mnist_multi':
        exp_data = torch.load('experiment_data/linear/multi/mnist_images.pt')
        exp_labels = torch.load('experiment_data/linear/multi/mnist_labels.pt')
        cuda = False

    #TODO Remove
    exp_data = exp_data[:3]
    exp_labels = exp_labels[:3]

    models = load_models(args.exp_type)

    log.info("Num Points {}".format(exp_data.size()[0]))
    log.info('Num Classifiers {}'.format(len(models)))

    if args.noise_function == 'pgd':
        adversary = partial(attacks.pgd, iters=args.pgd_iters, cuda=cuda)
    else:
        if args.exp_type == 'mnist_binary':
            adversary = attacks.distributional_oracle_binary
        elif args.exp_type == 'mnist_multi':
            adversary = attacks.distributional_oracle_multi
        for model in models:
            model.oracle = True

    noise_vectors, weights, expected_losses, minimum_losses = run_mwu(models, args.mwu_iters, exp_data, exp_labels,
                                                                      args.noise_budget, adversary, cuda)
    torch.save(noise_vectors, exp_dir + '/noise_vectors.pt')

    np.save(exp_dir  + "/weights.npy", weights)
    np.save(exp_dir  + "/expected_losses.npy", expected_losses)
    np.save(exp_dir  + "/minimum_losses.npy", minimum_losses)

    log.info("Finished!")

if __name__ == "__main__":
    main(sys.argv[1:])
