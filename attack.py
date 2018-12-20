import torch 

def pgd(weights, models, x, y, noise_budget, iters):
    step_size = noise_budget / (.8 * iters)
    noise_vector = torch.zeros(x.size()).cuda()
    #loss_list = []
    curr_x = x
    for i in range(iters):
        var_x = torch.autograd.Variable(curr_x, requires_grad=True).cuda()
        grad = torch.zeros(x.size()).cuda()
        #total_loss = 0
        for w, model in zip(weights, models):

            if var_x.grad is not None:
                var_x.grad.data.zero_()

            loss = w * model.loss_single(var_x, y)
            #total_loss += loss

            loss.backward()

            grad += var_x.grad.data
        #loss_list.append(total_loss)

        grad_norm = grad.norm(2)
        if grad_norm > 0:
            noise_vector += -1 * step_size * grad / grad.norm()
            noise_norm = torch.norm(noise_vector, p=2)

            if  noise_norm > noise_budget:
                noise_vector = noise_budget * noise_vector / noise_norm

            curr_x = torch.clamp(x + noise_vector, min=0.0, max=1.0)
        else:
            break
    return noise_vector #, loss_list
