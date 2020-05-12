import torch
import Node


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
        self.acc_best = torch.zeros(self.args.node_num + 1)
        self.get_a_better = torch.zeros(self.args.node_num + 1)

    def validate(self, node):
        self.counter += 1
        node.model.to(node.device).eval()
        total_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for idx, (data, target) in enumerate(node.test_data):
                data, target = data.to(node.device), target.to(node.device)
                output = node.model(data)
                total_loss += torch.nn.CrossEntropyLoss()(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss = total_loss / (idx + 1)
            acc = correct / len(node.test_data.dataset) * 100
        self.val_loss[str(node.num)].append(total_loss)
        self.val_acc[str(node.num)].append(acc)

        if self.val_acc[str(node.num)][-1] > self.acc_best[node.num]:
            self.get_a_better[node.num] = 1
            self.acc_best[node.num] = self.val_acc[str(node.num)][-1]
            torch.save(node.model.state_dict(),
                       './saves/model/Node{:d}_{:s}.pt'.format(node.num, node.args.local_model))

    def printer(self, node):
        if self.get_a_better[node.num] == 1:
            print('Node{:d}: A Better Accuracy: {:.2f}%! Model Saved!'.format(node.num, self.acc_best[node.num]))
            print('-------------------------')
            self.get_a_better[node.num] = 0

    def finish(self):
        torch.save([self.val_loss, self.val_acc],
                   './saves/record/loss_acc_{:s}_{:s}.pt'.format(self.args.algorithm, self.args.notes))
        print('Finished!\n')
        for i in range(self.args.node_num + 1):
            print('Node{}: Best Accuracy = {:.2f}%'.format(i, self.acc_best[i]))


def Catfish(Node_List, args):
    if args.catfish is None:
        pass
    else:
        Node_List[0].model = Node.init_model(args.catfish)
        Node_List[0].optimizer = Node.init_optimizer(Node_List[0].model, args)


def LR_scheduler(rounds, Node_List, args):
    trigger = int(args.R / 3)
    if rounds != 0 and rounds % trigger == 0 and rounds < args.stop_decay:
        args.lr *= 0.1
        # args.alpha += 0.2
        # args.beta += 0.4
        for i in range(len(Node_List)):
            Node_List[i].args.lr = args.lr
            Node_List[i].args.alpha = args.alpha
            Node_List[i].args.beta = args.beta
            Node_List[i].optimizer.param_groups[0]['lr'] = args.lr
            Node_List[i].meme_optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))


def Summary(args):
    print("Summary：\n")
    print("algorithm:{}\n".format(args.algorithm))
    print("dataset:{}\tbatchsize:{}\n".format(args.dataset, args.batchsize))
    print("node_num:{},\tsplit:{}\n".format(args.node_num, args.split))
    # print("iid:{},\tequal:{},\n".format(args.iid == 1, args.unequal == 0))
    print("global epochs:{},\tlocal epochs:{},\n".format(args.R, args.E))
    print("global_model:{}，\tlocal model:{},\n".format(args.global_model, args.local_model))
