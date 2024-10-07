from __init__ import *

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import timeit

import data.dataloader as dl
import strawml.models.straw_classifier.cnn_classifier as cnn
from strawml.models.straw_classifier.convnextv2 import *
import strawml.models.straw_classifier.chute_cropper as cc


def train_model(args, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> None:
    """Train the CNN classifier model.
    """
    
    if torch.cuda.is_available():
        print('Using GPU')
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.functional.cross_entropy
    
    
    for epoch in range(args.epochs):
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        train_iterator.set_description(f'Training Epoch {epoch+1}/{args.epochs}')
        model.train()
        epoch_accuracies = []
        # TODO: Temporary bbox this way (?)
        for (frame_data, target) in train_iterator:
            fullness = target
            
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

            # Save accuracy
            _, predicted = torch.max(output, 1)
            _, target_fullness = torch.max(fullness, 1)
            correct = sum(predicted == target_fullness)
            accuracy = 100 * correct / args.batch_size
            epoch_accuracies.append(accuracy.item())
        
            train_iterator.set_postfix(loss=loss.item(), accuracy=sum(epoch_accuracies)/len(epoch_accuracies))
        
        print(f'Epoch: {epoch+1}, Training Accuracy: {sum(epoch_accuracies)/len(epoch_accuracies):.2f}%')
        model.eval()
        
        
        with torch.no_grad():
            correct = 0
            total = 0
            val_iterator = tqdm(val_loader, unit="batch", position=0, leave=False)
            val_iterator.set_description(f'Validating Epoch {epoch+1}/{args.epochs}')
            
            # Time the inference time
            start_time = timeit.default_timer()
            
            for (frame_data, target) in val_iterator:
                fullness = target
                
                if torch.cuda.is_available():
                    frame_data = frame_data.cuda()
                    fullness = fullness.cuda()
                
                output = model(frame_data)
                _, predicted = torch.max(output, 1)
                total += args.batch_size
                _, target_fullness = torch.max(fullness, 1)
                correct += sum(predicted == target_fullness)
            
            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            average_time = elapsed_time / len(val_loader)
            
            accuracy = 100 * correct /total
            correct = correct.detach().cpu()
            
            print(f'Epoch: {epoch+1}, Validation Accuracy: {accuracy:.2f}%. Average Inference Time: {average_time:.6f} seconds, Total Inference Time: {elapsed_time:.2f} seconds. (Batch Size: {args.batch_size})')



def get_args():
    """Get the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train the CNN classifier model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--model_path', type=str, default='models/cnn_classifier.pth', help='Path to load the model from')
    parser.add_argument('--save_path', type=str, default='models/cnn_classifier_new.pth', help='Path to save the model to')
    parser.add_argument('--data_path', type=str, default='data/processed/augmented/chute_detection.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', type=bool, default=False, help='Include heatmaps in the training data')
    parser.add_argument('--inc_edges', type=bool, default=True, help='Include edges in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    image_size = (1370//2, 204//2)
    # image_size = (224, 224)
    
    train_set = dl.Chute(data_path=args.data_path, data_type='train', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges, random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size)
    test_set = dl.Chute(data_path=args.data_path, data_type='test', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges, random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    input_channels = 3
    if args.inc_heatmap: input_channels += 3
    if args.inc_edges: input_channels += 1
    model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels)
    # model = convnextv2_atto()
    
    train_model(args, model, train_loader, test_loader)

    
