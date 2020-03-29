import torch
from Train_mutual import train
from Validate import validate
from Node import Node, Global_Node
from torch.utils.tensorboard import SummaryWriter
from options import args_parser
from Data import Data

# init args
args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print("Running on", device)
data = Data(args)
logger = SummaryWriter('./logs')

# init nodes
global_node = Global_Node(args)
Node_List = []
for i in range(args.node_num):
    Node_List.append(Node(i + 1, data.train_loader, args))

# init variables
loss = 0
acc = 0
acc_best = 0
get_a_better = 0

# start
for rounds in range(args.R):
    print('==========The {:d}-th round=========='.format(rounds + 1))
    if rounds != 0 and rounds % args.lr_step == 0:
        args.lr = args.lr * 0.1
    print('Learning rate={:.4f}'.format(args.lr))
    for i in range(len(Node_List)):
        Node_List[i].fork(global_node)
        for epoch in range(args.E):
            train(Node_List[i])
            loss, acc = validate(Node_List[i], device, data.test_loader)
            logger.add_scalar('acc', acc)
            msg = "val_Loss = {:.4f} val_Accuracy = {:.2f}%\n".format(epoch + 1, loss, acc)
            if acc > acc_best:
                get_a_better = 1
                acc_best = acc
                torch.save(Node_List[i].model.state_dict(), "Node1_LeNet5.pt")
    if get_a_better == 1:
        print("A Better Accuracy: {:.2f}%! Model Saved!\n".format(acc_best))
        get_a_better = 0
    global_node.merge(Node_List)
print("Finished! The Best Accuracy: {:.2f}%! Model Saved!\n".format(acc_best))
