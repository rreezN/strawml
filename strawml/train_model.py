import argparse
import torch

import data.dataloader as dl
import models.cnn_classifier as cnn


def train_model(args, model, train_loader, val_loader):
    """Train the CNN classifier model.
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.functional.cross_entropy
    
    for epoch in args.epochs:
        model.train()
        for i, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
                
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in val_loader:
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
            print(f'Epoch: {epoch}, Validation Accuracy: {100*correct/total}')
            
    
    
    return


def get_args():
    """Get the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train the CNN classifier model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--model_path', type=str, default='models/cnn_classifier.pth', help='Path to load the model from')
    parser.add_argument('--save_path', type=str, default='models/cnn_classifier_new.pth', help='Path to save the model to')
    parser.add_argument('--data_path', type=str, default='data/processed/annotated_images.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', type=bool, default=True, help='Include heatmaps in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    train_loader = dl.Straw(data_path=args.data_path, data_type='train', batch_size=args.batch_size, inc_heatmap=args.inc_heatmap, seed=args.seed)
    test_loader = dl.Straw(data_path=args.data_path, data_type='test', batch_size=args.batch_size, inc_heatmap=args.inc_heatmap, seed=args.seed)
    
    cnn_model = cnn.CNNClassifier()
    
    train_model(args, cnn_model, train_loader, test_loader)


