from tqdm import tqdm
import torch.nn as nn

KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def train_normal(node):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(idx + 1, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100


def train_avg(node):
    node.meme.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.meme_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.meme(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.meme_optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
    node.model = node.meme


def train_mutual(node):
    node.model.to(node.device).train()
    node.meme.to(node.device).train()
    train_loader = node.train_data
    total_local_loss = 0.0
    avg_local_loss = 0.0
    correct_local = 0.0
    acc_local = 0.0
    total_meme_loss = 0.0
    avg_meme_loss = 0.0
    correct_meme = 0.0
    acc_meme = 0.0
    description = 'Node{:d}: loss_model={:.4f} acc_model={:.2f}% loss_meme={:.4f} acc_meme={:.2f}%'
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            node.meme_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_local_loss, acc_local, avg_meme_loss, acc_meme))
            data, target = data.to(node.device), target.to(node.device)
            output_local = node.model(data)
            output_meme = node.meme(data)
            ce_local = CE_Loss(output_local, target)
            kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))
            ce_meme = CE_Loss(output_meme, target)
            kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))
            loss_local = node.args.alpha * ce_local + (1 - node.args.alpha) * kl_local
            loss_meme = node.args.beta * ce_meme + (1 - node.args.beta) * kl_meme
            loss_local.backward()
            loss_meme.backward()
            node.optimizer.step()
            node.meme_optimizer.step()
            total_local_loss += loss_local
            avg_local_loss = total_local_loss / (idx + 1)
            pred_local = output_local.argmax(dim=1)
            correct_local += pred_local.eq(target.view_as(pred_local)).sum()
            acc_local = correct_local / len(train_loader.dataset) * 100
            total_meme_loss += loss_meme
            avg_meme_loss = total_meme_loss / (idx + 1)
            pred_meme = output_meme.argmax(dim=1)
            correct_meme += pred_meme.eq(target.view_as(pred_meme)).sum()
            acc_meme = correct_meme / len(train_loader.dataset) * 100


class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'fed_mutual':
            self.train = train_mutual
        elif args.algorithm == 'fed_avg':
            self.train = train_avg
        elif args.algorithm == 'normal':
            self.train = train_normal

    def __call__(self, node):
        self.train(node)
