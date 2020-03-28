import torch.nn.functional as F
from tqdm import tqdm


def train(node):
    node.model.to(node.device).train()
    train_loader = node.local_data
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
            pred = node.model(data)
            loss = F.cross_entropy(pred, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
