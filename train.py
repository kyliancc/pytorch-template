import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse

from dataset import TemplateDataset
from model import Net


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--lr', '-l', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='how many dataloader workers')
    parser.add_argument('--load', '-f', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--val', '-v', type=int, default=50, help='how many iterations to validate')
    parser.add_argument('--save', '-s', type=int, default=200, help='how many iterations to save')
    return parser.parse_args()


def val(loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for val_x, val_y in loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_pred = torch.argmax(model(val_x), dim=1)
            total += val_pred.size(0)
            correct += val_pred.eq(val_y).sum().item()

        accuracy = correct / total
        return accuracy


def main():
    args = parse_arguments()
    print(f'Arguments: {args.__dict__}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_dataset = TemplateDataset('train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = TemplateDataset('val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.load:
        state_dict = torch.load(args.load)
        model.iterations = state_dict['iterations']
        model.load_state_dict(state_dict['model'])
        model.optimizer.load_state_dict(state_dict['optimizer'])
        print(f'Loaded checkpoint from {args.load}.')

    print('START TRAINING')

    for epoch in range(args.epochs):
        for x, y in train_loader:
            model.train()
            x, y = x.to(device), y.to(device)

            model.optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            model.optimizer.step()

            print(f'Iteration {model.iterations} finished with loss: {loss.item()}')
            model.iterations += 1

            accuracy = None
            if model.iterations % args.val == 0:
                accuracy = val(val_loader, model, device)
                print(f'Validation finished with accuracy: {accuracy:.2%}')

            if model.iterations % args.save == 0:
                if accuracy is None:
                    accuracy = val(val_loader, model, device)
                    print(f'Validation finished with accuracy: {accuracy:.2%}')

                state_dict = {
                    'iterations': model.iterations,
                    'model': model.state_dict(),
                    'optimizer': model.optimizer.state_dict(),
                    'accuracy': accuracy
                }
                torch.save(state_dict, f'./checkpoints/template-net-{model.iterations}.pth')
                print(f'Checkpoint saved to ./checkpoints/template-net-{model.iterations}.pth')


if __name__ == '__main__':
    main()
