import argparse
import torch
from tqdm import tqdm

import data.dataloader as dl
import strawml.models.straw_classifier.cnn_classifier as cnn


def train_model(args, model: torch.nn.Module, train_loader: torch.utils.data.Dataset, val_loader: torch.utils.data.Dataset) -> None:
    """Train the CNN classifier model.
    """
    
    if torch.cuda.is_available():
        print('Using GPU')
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.functional.cross_entropy
    
    for epoch in range(args.epochs):
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        train_iterator.set_description(f'Epoch {epoch+1}/{args.epochs}')
        model.train()
        for (data, target) in train_iterator:
            
            frame_data = data[0]
            fullness = target[2]
            
            if torch.cuda.is_available():
                frame_data = frame_data.cuda()
                fullness = fullness.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(frame_data)
            loss = loss_fn(output, fullness)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_iterator.set_postfix(loss=loss.item())
                
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in val_loader:
                frame_data = data[0]
                fullness = target[2]
                
                if torch.cuda.is_available():
                    frame_data = frame_data.cuda()
                    fullness = fullness.cuda()
                
                output = model(frame_data)
                _, predicted = torch.max(output, 0)
                total += 1
                _, target_fullness = torch.max(fullness, 0)
                correct += (predicted == target_fullness)
                
            print(f'Epoch: {epoch+1}, Validation Accuracy: {100*correct/total:.2f}%')
            
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
    parser.add_argument('--data_path', type=str, default='data/processed/chute_detection.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', type=bool, default=True, help='Include heatmaps in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    train_loader = dl.Chute(data_path=args.data_path, data_type='train', inc_heatmap=args.inc_heatmap, random_state=args.seed)
    test_loader = dl.Chute(data_path=args.data_path, data_type='test', inc_heatmap=args.inc_heatmap, random_state=args.seed)
    
    cnn_model = cnn.CNNClassifier()
    
    train_model(args, cnn_model, train_loader, test_loader)


