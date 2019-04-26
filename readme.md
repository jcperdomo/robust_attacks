# Linear Experiments Setup Script
Setup Scripts:
python linear_experiments_setup.py -num_points 100 -num_classifiers 5 -exclusive False -sel_labels 0 1 2   
python linear_experiments_setup.py -num_points 100 -num_classifiers 5 -exclusive False -sel_labels 0 1    


# Code to execute experiments
Experiment Scripts:
Binary Oracle 
python main.py -exp_type mnist_binary -mwu_iters 30 -noise_function oracle -noise_budget 2.3  -name binary_oracle_1

Binary Grad
python main.py -exp_type mnist_binary -mwu_iters 30 -noise_function pgd  -noise_budget 2.3 -pgd_iters 40 -name binary_pgd_1

Multi Grad 
python main.py -exp_type mnist_multi -mwu_iters 30 -noise_function pgd  -noise_budget 1.3 -pgd_iters 40 -name multi_pgd_1

Multi Oracle 
python main.py -exp_type mnist_multi -mwu_iters 30 -noise_function oracle  -noise_budget 1.3  -name multi_pgd_1

Imagenet 
python main.py -exp_type imagenet -mwu_iters 30 -noise_function pgd  -noise_budget .8 -pgd_iters 40 -name imagenet_1


