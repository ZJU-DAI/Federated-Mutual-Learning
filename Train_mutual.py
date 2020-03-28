import torch.nn.functional as F
from tqdm import tqdm


def train(node):
    node.model.train()
    node.meme.train()
    train_loader = node.local_data
    total_local_loss = 0.0
    avg_local_loss = 0.0
    correct_local = 0.0
    acc_local = 0.0
    total_meme_loss = 0.0
    avg_meme_loss = 0.0
    correct_meme = 0.0
    acc_meme = 0.0
    description_local = "Node{:d}: loss_model={:.4f} acc_model={:.2f}% loss_meme={:.4f} acc_meme={:.2f}%"

    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            node.meme_optimizer.zero_grad()
            epochs.set_description(
                description_local.format(node.num, avg_local_loss, acc_local, avg_meme_loss, acc_meme))
            data, target = data.to(node.device), target.to(node.device)
            pred_local = node.model(data)
            pred_meme = node.meme(data)
            loss_local = F.kl_div(F.log_softmax(pred_local, dim=1), F.softmax(pred_meme.detach(), dim=1),
                                  reduction='batchmean') + F.cross_entropy(pred_local, target)
            loss_meme = F.kl_div(F.log_softmax(pred_meme, dim=1), F.softmax(pred_local.detach(), dim=1),
                                 reduction='batchmean') + F.cross_entropy(pred_meme, target)
            loss_local.backward()
            loss_meme.backward()
            node.optimizer.step()
            node.meme_optimizer.step()
            total_local_loss += loss_local
            avg_local_loss = total_local_loss / (idx + 1)
            pred_local = pred_local.argmax(dim=1)
            correct_local += pred_local.eq(target.view_as(pred_local)).sum()
            acc_local = correct_local / len(train_loader.dataset) * 100
            total_meme_loss += loss_meme
            avg_meme_loss = total_meme_loss / (idx + 1)
            pred_meme = pred_meme.argmax(dim=1)
            correct_meme += pred_meme.eq(target.view_as(pred_meme)).sum()
            acc_meme = correct_meme / len(train_loader.dataset) * 100
    node.scheduler.step()
    node.meme_scheduler.step()
