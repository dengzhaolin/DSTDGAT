import argparse
import torch
# ordered by first character
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--embed_size', type=int, default=64,
                    help='dims of embedding')
parser.add_argument('--DEVICE', default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help='computational device')
parser.add_argument('--drop', default= 0,
                    help='dropout of model')
parser.add_argument('--decay_epoch', type=int, default=5,
                    help='decay epoch')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of ST blocks')
parser.add_argument('--heads', type=int, default=2,
                    help='number of attention heads')
parser.add_argument('--log_file', default='./data/log',
                    help='log file')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--max_epoch', type=int, default=150,
                    help='epoch to run')
parser.add_argument('--model_file', default='./data/ST_PEMS.pkl',
                    help='save the model to disk')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_day', type=int, default=3,
                    help='daily steps')
parser.add_argument('--num_week', type=int, default=3,
                    help='weekly steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps: 5min/per')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')
parser.add_argument('--SE_file', default='./data/SE(PeMS).txt',
                    help='spatial embedding file')
parser.add_argument('--shuffle', default=True, help='shuffle of train dataloader')
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--traffic_file', default='./data/pems-bay.h5',
                    help='traffic file')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--gamma', type=float, default=0.5)

args = parser.parse_args()