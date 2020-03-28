import torch.nn.functional as F
from tqdm import tqdm
import torch
import Model

teacher = Model.ResNet18()
teacher.load_state_dict(torch.load("Teacher_ResNet18.pt"))


def train(node, device, optimizer):
    teacher.to(device)
    model1 = node.model.to(device)
    train_loader = node.local_data
    model1.train()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    T = 6
    description = "Training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"

    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            optimizer.zero_grad()
            epochs.set_description(description.format(idx + 1, avg_loss, acc))
            data, target = data.to(device), target.to(device)
            soft_label = F.softmax(teacher(data) / T, dim=1)
            pred = model1(data)
            loss1 = torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(pred / T, dim=1),
                                                              soft_label) * 0.5 * T * T + F.cross_entropy(pred,
                                                                                                          target) * 0.5
            loss1.backward()
            optimizer.step()
            total_loss += loss1
            avg_loss = total_loss / (idx + 1)
            pred = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
