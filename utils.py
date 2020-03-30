def LR_scheduler(rounds, Node_List, args):
    if rounds != 0 and rounds % args.lr_step == 0:
        args.lr = args.lr * 0.1
        for i in range(len(Node_List)):
            Node_List[i].args.lr = args.lr
            Node_List[i].optimizer.param_groups[0]['lr'] = args.lr
            Node_List[i].meme_optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))
