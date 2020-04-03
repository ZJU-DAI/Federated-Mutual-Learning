import torch.nn.functional as F
from tqdm import tqdm


def train(node):
    node.meme.to(node.device).train()
    train_loader = node.local_data
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
            loss = F.cross_entropy(output, target)
            loss.backward()
            node.meme_optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
    node.model = node.meme
    # return avg_loss, acc
