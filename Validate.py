import torch


def validate(node, device, test_loader):
    node.model.to(device).eval()
    total_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = node.model(data)
            total_loss = total_loss + torch.nn.CrossEntropyLoss()(output, target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss = total_loss / (idx + 1)
        acc = correct / len(test_loader.dataset) * 100
    return total_loss, acc
