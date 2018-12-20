import torch.nn as nn
import torch

class DNN(nn.Module):
    def __init__(self, model, cuda=True):
        super(DNN, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.model = model
        if cuda:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
            self.var = torch.tensor([0.229, 0.224, 0.225]).pow(2).cuda()
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406])
            self.var = torch.tensor([0.229, 0.224, 0.225]).pow(2)

    def forward(self, x):
        x = nn.functional.batch_norm(x, self.mean, self.var, momentum=0.0, eps=0.0)
        logits = self.model(x)
        return self.softmax(logits)

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

    def accuracy(self, X, Y):
        out = self.forward(X)
        _, preds = torch.max(out, 1)
        return (preds == Y).to(torch.float).mean().item()

    def loss_single(self, x, y):
        probs = self.forward(x)
        true_max, max_ix = torch.max(probs, 1)
        probs2 = probs.clone()
        probs2[0, y.item()] = -1.0
        _, second_max_ix = torch.max(probs2,1)
        relu = nn.ReLU()
        return relu(probs[0, y.item()] - probs[0, second_max_ix.item()])
