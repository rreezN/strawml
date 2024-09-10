import argparse
import torch
from tqdm import tqdm

import data.dataloader as dl
import models.cnn_classifier as cnn


def train_model(args, model, train_loader, val_loader):
    """Train the CNN classifier model.
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.functional.cross_entropy
    
    for epoch in range(args.epochs):
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        train_iterator.set_description(f'Epoch {epoch+1}/{args.epochs}')
        model.train()
        for (data, target) in train_iterator:
            
            frame_data = data[0]
            fullness = target[3]
            
            # Forward pass
            output = model(frame_data)
            loss = loss_fn(output, torch.Tensor(fullness))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_iterator.set_postfix(loss=loss.item())
                
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in val_loader:
                frame_data = data[0]
                fullness = target[3]
                
                output = model(frame_data)
                _, predicted = torch.max(output, 0)
                total += 1
                correct += (predicted == target)
                
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
    
    train_loader = dl.Straw(data_path=args.data_path, data_type='train', inc_heatmap=args.inc_heatmap, random_state=args.seed)
    test_loader = dl.Straw(data_path=args.data_path, data_type='test', inc_heatmap=args.inc_heatmap, random_state=args.seed)
    
    cnn_model = cnn.CNNClassifier()
    
    train_model(args, cnn_model, train_loader, test_loader)


