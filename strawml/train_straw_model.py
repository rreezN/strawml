from __init__ import *

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timeit
import timm
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np

import data.dataloader as dl
import strawml.models.straw_classifier.cnn_classifier as cnn
import strawml.models.straw_classifier.feature_model as feature_model


def train_model(args, model: torch.nn.Module, train_loader: DataLoader, val_loader: torch.utils.data.DataLoader, feature_regressor: torch.nn.Module = None) -> None:
    """Train the CNN classifier model.
    """
    
    if torch.cuda.is_available():
        print('Using GPU')
        model = model.cuda()
        if feature_regressor is not None:
            feature_regressor = feature_regressor.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.cont:
        loss_fn = torch.nn.functional.mse_loss
        best_accuracy = 1000000.0
    else:
        loss_fn = torch.nn.functional.cross_entropy
        best_accuracy = 0.0
    
    plot_examples_every = 50 # Save an example every X% of the length of the training data
    for epoch in range(args.epochs):
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        train_iterator.set_description(f'Training Epoch {epoch+1}/{args.epochs}')
        model.train()
        if feature_regressor is not None:
            feature_regressor.train()
        epoch_accuracies = []
        epoch_losses = []
        current_iteration = 0
        for (frame_data, target) in train_iterator:
            current_iteration += 1
            # TRY: using only the edge image
            # frame_data = frame_data[:, 3, :, :]
            # frame_data = frame_data.unsqueeze(1)

            fullness = target
            
            if torch.cuda.is_available():
                frame_data = frame_data.cuda()
                fullness = fullness.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.cont and args.model != 'cnn':
                features = model.forward_features(frame_data)
                output = features
            else:
                output = model(frame_data)
            if args.cont and len(output.shape) > 3: output = output.squeeze()
            if feature_regressor is not None:
                output = feature_regressor(output)
            if args.cont:
                if len(output.shape) > 1:
                    output = output.flatten()
                # output = torch.clamp(output, 0, 1)
            
            loss = loss_fn(output, fullness)
            
            epoch_losses += [loss.item()]
            
            # Backward pass
            loss.backward()
            optimizer.step()

            if args.cont:
                train_iterator.set_postfix(loss=loss.item(), fullness=torch.mean(fullness).item(), prediction=torch.mean(output).item())
            else:
            # Save accuracy
                _, predicted = torch.max(output, 1)
                _, target_fullness = torch.max(fullness, 1)
                correct = sum(predicted == target_fullness)
                accuracy = 100 * correct / args.batch_size
                epoch_accuracies.append(accuracy.item())
        
                train_iterator.set_postfix(loss=loss.item(), accuracy=sum(epoch_accuracies)/len(epoch_accuracies))
            
            if not args.no_wandb:
                # Save an example every 10% of the length of the training data
                training_info = {'data_type': 'train', 'current_iteration': current_iteration, 'epoch': epoch+1}
                divisor = int(len(train_loader)/plot_examples_every)
                divisor = 1 if divisor == 0 else divisor
                if current_iteration % divisor == 0:
                    if args.cont:
                        plot_example(training_info, frame_data, output, fullness)
                    else:
                        plot_example(training_info, frame_data, predicted, target_fullness)

        if args.cont:
            print(f'Epoch: {epoch+1}, Training Loss: {sum(epoch_losses)/len(epoch_losses)}, Last predictions -- Fullness: {torch.mean(fullness).item()}, Prediction: {torch.mean(output).item()}')                
        else:
            print(f'Epoch: {epoch+1}, Training Accuracy: {sum(epoch_accuracies)/len(epoch_accuracies):.2f}%')
        
        model.eval()
        if feature_regressor is not None:
            feature_regressor.eval()
        
        with torch.no_grad():
            correct = 0
            total = 0
            val_iterator = tqdm(val_loader, unit="batch", position=0, leave=False)
            val_iterator.set_description(f'Validating Epoch {epoch+1}/{args.epochs}')
            
            # Time the inference time
            start_time = timeit.default_timer()
            
            val_lossses = []
            current_iteration = 0
            batch_times = []
            for (frame_data, target) in val_iterator:
                current_iteration += 1
                # TRY: using only the edge image
                # frame_data = frame_data[:, 3, :, :]
                # frame_data = frame_data.unsqueeze(1)
                
                fullness = target
                
                if torch.cuda.is_available():
                    frame_data = frame_data.cuda()
                    fullness = fullness.cuda()
                
                
                # Forward pass
                if args.cont and args.model != 'cnn':
                    features = model.forward_features(frame_data)
                    output = features
                else:
                    output = model(frame_data)
                if args.cont: output = output.squeeze()
                if feature_regressor is not None:
                    output = feature_regressor(output)
                    output = output.squeeze()
                # if args.cont:
                #     output = torch.clamp(output, 0, 1)
                
                batch_time = timeit.default_timer() - start_time
                batch_time = batch_time/frame_data.shape[0]
                batch_times.append(batch_time)
                
                val_loss = loss_fn(output, fullness)
                val_lossses.append(val_loss.item())
                    
                if not args.cont:
                    output = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    total += args.batch_size
                    _, target_fullness = torch.max(fullness, 1)
                    correct += sum(predicted == target_fullness)

                
                if not args.no_wandb:
                    # Save an example every 10% of the length of the training data
                    val_info = {'data_type': 'val', 'current_iteration': current_iteration, 'epoch': epoch+1}
                    divisor = int(len(train_loader)/plot_examples_every)
                    divisor = 1 if divisor == 0 else divisor
                    if current_iteration % divisor == 0:
                        if args.cont:
                            plot_example(val_info, frame_data, output, fullness)
                        else:
                            plot_example(val_info, frame_data, predicted, target_fullness)
                
            average_time = sum(batch_times) / len(batch_times)
            
            if args.cont:
                print(f'Epoch: {epoch+1}, Average Inference Time: {average_time:.6f} Validation Loss: {sum(val_lossses)/len(val_lossses)}, Last predictions -- Fullness: {torch.mean(fullness).item()}, Prediction: {torch.mean(output).item()}')
                if sum(val_lossses)/len(val_lossses) < best_accuracy:
                    best_accuracy = sum(val_lossses)/len(val_lossses)
                    print(f'New best loss: {best_accuracy}')
                    if args.model != 'cnn':
                        model_folder = f'{args.save_path}{args.model}_regressor/'
                        os.makedirs(model_folder, exist_ok=True)
                        model_name = args.model + '_feature_extractor'
                        model_save_path = model_folder + model_name + '_best.pth'
                        regressor_name = args.model + '_regressor'
                        regressor_save_path = model_folder + regressor_name + '_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                        torch.save(feature_regressor.state_dict(), regressor_save_path)
                        if not args.no_wandb:
                            wandb.save(model_save_path)
                            wandb.save(regressor_save_path)
                    else:
                        model_name = args.model + '_regressor'
                        model_save_path = args.save_path + model_name + '_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                        if not args.no_wandb:
                            wandb.save(model_save_path)
                            
                    if not args.no_wandb:
                        wandb.log(step=epoch+1, data={'best_val_loss': best_accuracy})
            else:
                accuracy = 100 * correct /total
                correct = correct.detach().cpu()
                print(f'Epoch: {epoch+1}, Validation Accuracy: {accuracy:.2f}%. Average Inference Time: {average_time:.6f} seconds, Total Inference Time: {sum(batch_times):.2f} seconds. (Batch Size: {args.batch_size})')
                if not args.no_wandb:
                    wandb.log(step=epoch+1, 
                              data={'train_accuracy': sum(epoch_accuracies)/len(epoch_accuracies), 
                                'val_accuracy': accuracy})
                
                if accuracy > best_accuracy:
                    print(f'New best accuracy: {accuracy:.2f}%')
                    best_accuracy = accuracy
                    model_name = args.model + '_classifier'
                    model_save_path = args.save_path + model_name + '_best.pth'
                    torch.save(model.state_dict(), model_save_path)
                    if not args.no_wandb:
                        wandb.log(step=epoch+1, data={'best_val_accuracy': best_accuracy})
                        wandb.save(model_save_path)
                    
            if not args.no_wandb:
                wandb.log(step=epoch+1, 
                        data={'train_loss': sum(epoch_losses)/len(epoch_losses), 
                              'val_loss': sum(val_lossses)/len(val_lossses), 
                              'epoch': epoch+1,
                              'inference_time': average_time,})
            
    
    if not args.no_wandb: wandb.finish()


def plot_example(info, frame_data, prediction, target):
    """Plot an example frame with the prediction.
    """
    if len(frame_data.shape) == 4:
        frame_data = frame_data[0]
    if len(prediction) > 1:
        prediction = prediction[0].detach().cpu().numpy()
    if len(target) > 1:
        target = target[0].detach().cpu().numpy()
    if frame_data.shape[0] > 3:
        frame_data = frame_data[:3]
    
    frame_data = frame_data.permute(1, 2, 0)
    frame_data = frame_data.detach().cpu().numpy()
    
    means = train_set.train_mean
    stds = train_set.train_std
    frame_data = frame_data * stds + means
    frame_data = np.clip(frame_data, 0, 1)
    
    if args.cont:
        prediction = round(prediction*100)
        target = round(target*100)
    else:
        increment = increment = 100 / (train_set.num_classes_straw - 1)
        prediction = prediction * increment
        target = target * increment
    
    plt.imshow(frame_data)
    plt.title(f'{info["data_type"]} Epoch: {info["epoch"]} it: {info["current_iteration"]} Prediction: {prediction} Target: {target}')
    
    # plt.show()
    
    if not args.no_wandb:
        wandb.log({f'{info["data_type"]}_example_frame': wandb.Image(plt)})
        
    plt.close()
    

def initialize_wandb(args: argparse.Namespace) -> None:
    """Initialize the Weights and Biases logging.
    """
    wandb.login()
    wandb.init(
        project='testrun',
        entity='meliora',
        config={
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'inc_heatmap': args.inc_heatmap,
            'inc_edges': args.inc_edges,
            'seed': args.seed,
            'model': args.model,
            'image_size': image_size,
            'num_classes_straw': args.num_classes_straw,
            'continuous': args.cont,
            'data_subsample': args.data_subsample
        })


def get_args():
    """Get the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train the CNN classifier model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for training')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the model to')
    parser.add_argument('--data_path', type=str, default='data/processed/augmented/chute_detection.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', type=bool, default=False, help='Include heatmaps in the training data')
    parser.add_argument('--inc_edges', type=bool, default=True, help='Include edges in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_wandb', action='store_true', help='Do not use Weights and Biases for logging')
    parser.add_argument('--model', type=str, default='vit', help='Model to use for training', choices=['cnn', 'convnextv2', 'vit', 'eva02', 'caformer'])
    parser.add_argument('--image_size', type=tuple, default=(1370, 204), help='Image size for the model (only for CNN)')
    parser.add_argument('--num_classes_straw', type=int, default=11, help='Number of classes for the straw classifier (11 = 10%, 21 = 5%)')
    parser.add_argument('--cont', action='store_true', help='Set model to predict a continuous value instead of a class')
    parser.add_argument('--data_subsample', type=float, default=1.0, help='Amount of the data to subsample for training (1.0 = 100%, 0.5 = 50%)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    match args.model:
        case 'cnn':
            image_size = args.image_size
        case 'convnextv2':
            image_size = (224, 224)
        case 'vit':
            image_size = (384, 384)
        case 'eva02':
            image_size = (448, 448)
    
    train_set = dl.Chute(data_path=args.data_path, data_type='train', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
                         random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
                         num_classes_straw=args.num_classes_straw, continuous=args.cont, subsample=args.data_subsample)
    test_set = dl.Chute(data_path=args.data_path, data_type='test', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
                        random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
                        num_classes_straw=args.num_classes_straw, continuous=args.cont, subsample=args.data_subsample)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    input_channels = 3
    if args.inc_heatmap: input_channels += 3
    if args.inc_edges: input_channels += 1
    
    # TRY: Using only the edge image as input
    # input_channels = 1
    
    # WANDB
    if not args.no_wandb:
        initialize_wandb(args)
    
    if args.cont:
        args.num_classes_straw = 1
    
    match args.model:
        case 'cnn':
            model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=args.num_classes_straw)
        case 'convnextv2':
            model = timm.create_model('convnext_small.in12k_ft_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=True)
        case 'vit':
            model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=True)
        case 'eva02':
            model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=True)
        case 'caformer':
                model = timm.create_model('caformer_m36,.sail_in22k_ft_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=True)
    
    if args.cont and args.model != 'cnn':
        feature_shape = model.forward_features(torch.randn(1, input_channels, image_size[0], image_size[1])).shape
        feature_size = feature_shape[1] * feature_shape[2]
        feature_regressor = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1)
    else:
        feature_regressor = None
    
    train_model(args, model, train_loader, test_loader, feature_regressor)

    