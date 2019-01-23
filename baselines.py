import numpy as np

def compute_model_accs_per_point(models, images, noise_vectors, labels):
    model_accs_per_point = []
    num_points = len(images)
    for i in range(num_points):
        x = images[i].unsqueeze(0)
        v = noise_vectors[i].unsqueeze(0)
        y = labels[i]
        model_accs_per_point.append([model.accuracy(x + v, y) for model in models])
    return np.array(model_accs_per_point)

def compute_best_individual_baseline(models, images, individual_noise_vectors, labels):
    individual_accs_per_model = np.array([compute_model_accs_per_point(models, images, 
                                                                       individual_noise_vectors[i], labels)
                                        for i in range(len(models))])
    mean_max = []
    for summary in [np.mean, np.max]:
        overall = []
        for i in range(len(images)):
            point_accuracy_per_model = []
            for m in range(len(models)):
                r = summary(individual_accs_per_model[m][i])
                point_accuracy_per_model.append(r)
            overall.append(min(point_accuracy_per_model))
        mean_max.append(np.mean(overall))
    return mean_max

