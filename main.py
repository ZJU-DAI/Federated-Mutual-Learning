import torch
import torchvision
from torchvision import transforms
from tqdm import trange
import Model
from Train_mutual import train
from Validate import validate
from Node import Node, Global_Node
from torch.utils.tensorboard import SummaryWriter
from options import args_parser

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Running on", device)
logger = SummaryWriter('./logs')

tra_transformer = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
val_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(
    root="~/data", train=True, download=False, transform=tra_transformer
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True, num_workers=4
)
test_set = torchvision.datasets.CIFAR10(
    root="~/data", train=False, download=False, transform=val_transformer
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False, num_workers=4
)

# args = args_parser()
global_node = Global_Node(Model.LeNet5(), device)
# node1 = Node(Model.CNNCifar(), train_loader, device)
# node2 = Node(Model.ResNet18(), train_loader, device)
loss = 0
acc = 0
acc_best = 0
get_a_better = 0
node_num = 2
Node_List = []
for i in range(node_num):
    exec('node_%d=Node(%d, Model.LeNet5(), train_loader, device)\n'
         'Node_List.append(node_%d)' % (i + 1, i + 1, i + 1))
R = 10
E = 1
for rounds in range(R):
    print('==========The {:d}-th round=========='.format(rounds + 1))
    for i in range(len(Node_List)):
        Node_List[i].fork(global_node)
        for epoch in range(E):
            # description = \
            #     "Node {:d}: Loss = {:.4f} Accuracy = {:.2f}%".format(i, epoch + 1, loss, acc)
            # E.set_description(description)
            train(Node_List[i])
            loss, acc = validate(Node_List[i], device, test_loader)
            logger.add_scalar('acc', acc)
            msg = "val_Loss = {:.4f} val_Accuracy = {:.2f}%\n".format(
                epoch + 1, loss, acc
            )
            # E.write(msg)
            if acc > acc_best:
                get_a_better = 1
                acc_best = acc
                torch.save(Node_List[i].model.state_dict(), "Node1_LeNet5.pt")
    if get_a_better == 1:
        print("A Better Accuracy: {:.2f}%! Model Saved!\n".format(acc_best))
        get_a_better = 0
    global_node.merge(Node_List)
print("Finished! The Best Accuracy: {:.2f}%! Model Saved!\n".format(acc_best))
