import argparse
import torch
from data import dataprocess,data_fluid
from models.Baseline import BaselineNetwork
from train import train
from inference import inference

def arg_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate BaselineNetwork")
    parser.add_argument('--data_path', type=str, default='data/data.xlsx')
    parser.add_argument('-history_length', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=13)
    parser.add_argument('--input_dim', type=int, default=120)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--output_path', type=str, default='/baksv/CIGIT/GXN_Liuxy/SSG/fluid/ckpts')
    return parser

def main():
    parser = arg_parser()
    args = parser.parse_args()

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    train_loader, val_loader, test_loader =dataprocess.get_dataloader(args)

    model = BaselineNetwork(args.input_dim, args.embed_dim, args.num_heads, args.hidden_dim, args.num_classes)

    if args.mode == 'train':
        print("Training model...")
        print(args)
        train(model, train_loader, val_loader, args)
    else:
        model.load_state_dict(torch.load('model.pth'))
        predictions = inference(model, test_loader, args)
        print(predictions)


if __name__ == "__main__":
    main()
