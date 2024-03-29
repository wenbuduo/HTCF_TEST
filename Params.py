import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='rt')  # rt tp
    parser.add_argument('--model', type=str, default='HTCF')  # NeuCF, CSMF, GraphMF, GATCF,HTCF

    # Experiment
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=1)#1显示进度条
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)#这里表示训练多少次显示一下结果
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--device', type=str, default='cuda')  # cuda gpu cpu mps
    parser.add_argument('--bs', type=int, default=1024)  # NeuCF 256 CSMF 256 GraphMF 256 GATCF 4096
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=50)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--saved', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')

    # Hyper parameters
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--windows', type=int, default=5)

    # NeuCF
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # GraphMF
    parser.add_argument('--order', type=int, default=2)
    # GATCF
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--heads', type=int, default=2)

    # Other Experiment
    parser.add_argument('--ablation', type=int, default=0)

    #HTCF
    parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')#这个需要测试
    parser.add_argument('--gcn_hops', default=2, type=int, help='number of hops in gcn precessing')
    return parser.parse_args()

args = parse_args()
# 根据训练数据集大小和批次大小来设置学习率衰减的步长
# trnNum 代表训练样本的总数或训练步数,batch代表每个训练批次中的样本数
#args.decay_step = args.trnNum // args.batch