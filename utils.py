import torch


def LR_scheduler(rounds, Node_List, args):
    if rounds != 0 and rounds % args.lr_step == 0 and rounds < args.stop_decay:
        args.lr = args.lr * 0.1
        args.alpha -= 0.2
        args.beta += 0.4
        if args.alpha < 0.5:
            args.alpha = 0.5
        for i in range(len(Node_List)):
            Node_List[i].args.lr = args.lr
            Node_List[i].args.alpha = args.alpha
            Node_List[i].args.beta = args.beta
            Node_List[i].optimizer.param_groups[0]['lr'] = args.lr
            Node_List[i].meme_optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))


class Recorder(object):
    def __init__(self, args):
        self.args = args
        self.counter = 0
        self.tra_loss = {}
        self.tra_acc = {}
        self.val_loss = {}
        self.val_acc = {}
        for i in range(self.args.node_num + 1):
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
        # self.loss = torch.zeros(self.args.node_num + 1)
        # self.acc = torch.zeros(self.args.node_num + 1)
        self.acc_best = torch.zeros(self.args.node_num + 1)
        self.get_a_better = torch.zeros(self.args.node_num + 1)
        # self.loss_meme = torch.zeros(args.node_num)
        # self.acc_meme = torch.zeros(args.node_num)
        # self.acc_best_meme = torch.zeros(args.node_num)
        # self.get_a_better_meme = torch.zeros(args.node_num)

    def train(self):
        # self.counter += 1
        pass

    def validate(self, node, test_loader):
        self.counter += 1
        node.model.to(node.device).eval()
        total_loss = 0.0
        correct = 0.0
        # node.meme.to(node.device).eval()
        # total_loss_meme = 0.0
        # correct_meme = 0.0
        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(node.device), target.to(node.device)
                output = node.model(data)
                total_loss += torch.nn.CrossEntropyLoss()(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                # output_meme = node.model(data)
                # total_loss_meme += torch.nn.CrossEntropyLoss()(output_meme, target)
                # pred_meme = output_meme.argmax(dim=1)
                # correct_meme += pred_meme.eq(target.view_as(pred_meme)).sum().item()
            total_loss = total_loss / (idx + 1)
            acc = correct / len(test_loader.dataset) * 100
            # total_loss_meme = total_loss_meme / (idx + 1)
            # acc_meme = correct_meme / len(test_loader.dataset) * 100
        self.val_loss[str(node.num)].append(total_loss)
        self.val_acc[str(node.num)].append(acc)
        # self.loss_meme[node.num] = total_loss_meme
        # self.acc_meme[node.num] = acc_meme
        if self.val_acc[str(node.num)][-1] > self.acc_best[node.num]:
            self.get_a_better[node.num] = 1
            self.acc_best[node.num] = self.val_acc[str(node.num)][-1]
            torch.save(node.model.state_dict(), "Node{:d}_{:s}.pt".format(node.num, node.args.local_model))

    def printer(self, node):
        if self.get_a_better[node.num] == 1:
            print("Node{:d}: A Better Accuracy: {:.2f}%! Model Saved!".format(node.num, self.acc_best[node.num]))
            print('-------------------------')
            self.get_a_better[node.num] = 0

    def finish(self):
        torch.save(self.val_loss,
                   '0.2dval_loss_{:s}.pt'.format(self.args.algorithm + str(self.args.alpha) + str(self.args.beta)))
        torch.save(self.val_acc,
                   '0.2dval_acc_{:s}.pt'.format(self.args.algorithm + str(self.args.alpha) + str(self.args.beta)))
        print("Finished!\n")
        for i in range(self.args.node_num + 1):
            print("Node{}: Best Accuracy = {:.2f}%".format(i, self.acc_best[i]))
