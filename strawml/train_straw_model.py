from __init__ import *

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
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
    
    match args.optim:
        case 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            if feature_regressor is not None:
                optimizer_feature = torch.optim.Adam(feature_regressor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        case 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if feature_regressor is not None:
                optimizer_feature = torch.optim.AdamW(feature_regressor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        case 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            if feature_regressor is not None:
                optimizer_feature = torch.optim.SGD(feature_regressor.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        case 'soap':
            raise NotImplementedError
        case 'adopt':
            raise NotImplementedError
    
    if args.cont:
        loss_fn = torch.nn.functional.mse_loss
        best_accuracy = 1000000.0
    else:
        if args.use_wce:
            ce_weights = torch.ones(args.num_classes_straw)
            # TODO: Update weights based on data distribution
            ce_weights[0, 1, 2] = 2.0
            ce_weights[-3, -2, -1] = 2.0
        loss_fn = torch.nn.functional.cross_entropy
        best_accuracy = 0.0
    
    plot_examples_every = 0.5 # Save an example every X% of the length of the training data
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
            if feature_regressor is not None: optimizer_feature.zero_grad()
            
            # Forward pass
            if args.cont and args.model != 'cnn':
                features = model.forward_features(frame_data)
                output = features
            else:
                output = model(frame_data)
            # print("features shape:", features.shape)
            # print("output shape (post squeeze, before feature_regressor):", output.shape)
            if feature_regressor is not None:
                output = output.flatten(1)
                output = feature_regressor(output)
                # output = torch.clamp(output, 0, 1)
            
            if args.use_wce:
                loss = loss_fn(output, fullness, weight=ce_weights)
            else:
                loss = loss_fn(output, fullness)
            
            epoch_losses += [loss.item()]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            if feature_regressor is not None: optimizer_feature.step()

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
           
            # TODO: For some reason this completely fucks up the wandb logging (something to do with steps out of order) 
            # if not args.no_wandb:
            #     # Save an example every 10% of the length of the training data
            #     training_info = {'data_type': 'train', 'current_iteration': current_iteration, 'epoch': epoch+1}
            #     divisor = int(len(train_loader) * plot_examples_every)
            #     divisor = 1 if divisor == 0 else divisor
            #     if current_iteration % divisor == 0:
            #         if args.cont:
            #             plot_example(training_info, frame_data, output, fullness)
            #         else:
            #             plot_example(training_info, frame_data, predicted, target_fullness)

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
                
                # Time the inference time
                start_time = timeit.default_timer()
                
                # Forward pass
                if args.cont and args.model != 'cnn':
                    features = model.forward_features(frame_data)
                    output = features
                else:
                    output = model(frame_data)
                # print("features shape:", features.shape)
                # print("output shape (post squeeze, before feature_regressor):", output.shape)
                if feature_regressor is not None:
                    output = output.flatten(1)
                    output = feature_regressor(output)
                    # output = torch.clamp(output, 0, 1)
                
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
                    divisor = int(len(train_loader) * plot_examples_every)
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
                    id = f'_{args.id}' if args.id != '' else ''
                    if args.model != 'cnn':
                        model_folder = f'{args.save_path}{args.model}_regressor/'
                        os.makedirs(model_folder, exist_ok=True)
                        model_name = args.model + f'_feature_extractor{id}'
                        model_save_path = model_folder + model_name + '_best.pth'
                        regressor_name = args.model + f'_regressor{id}'
                        regressor_save_path = model_folder + regressor_name + '_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                        torch.save(feature_regressor.state_dict(), regressor_save_path)
                        if not args.no_wandb:
                            wandb.save(model_save_path)
                            wandb.save(regressor_save_path)
                    else:
                        model_name = args.model + f'_regressor{id}'
                        model_save_path = args.save_path + model_name + '_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                        if not args.no_wandb:
                            wandb.save(model_save_path)
                            
                    if not args.no_wandb:
                        wandb.log(step=epoch+1, data={'best_val_loss': best_accuracy})
            else:
                accuracy = 100 * correct /total
                correct = correct.detach().cpu()
                print(f'Epoch: {epoch+1}, Validation Accuracy: {accuracy:.2f}%. Average Inference Time: {average_time:.6f} seconds, Total Inference Time: {sum(batch_times):.6f} seconds. (Batch Size: {args.batch_size})')
                if not args.no_wandb:
                    wandb.log(step=epoch+1, 
                              data={'train_accuracy': sum(epoch_accuracies)/len(epoch_accuracies), 
                                'val_accuracy': accuracy})
                
                if accuracy > best_accuracy:
                    print(f'New best accuracy: {accuracy:.2f}%')
                    best_accuracy = accuracy
                    id = f'_{args.id}' if args.id != '' else ''
                    model_folder = f'{args.model}_classifier/'
                    model_name = args.model + f'_classifier{id}'
                    model_save_path = args.save_path + model_folder + model_name + '_best.pth'
                    os.makedirs(f'{args.save_path}{model_folder}', exist_ok=True)
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
            

def predict_sensor_data(args, model: torch.nn.Module, sensor_loader: DataLoader, feature_regressor: torch.nn.Module = None):
    """Predict the output of a model on a dataset.
    """
    
    print("Predicting Sensor Data")
    if torch.cuda.is_available():
        model = model.cuda()
        if feature_regressor is not None:
            feature_regressor = feature_regressor.cuda()
    
    if args.cont:
        loss_fn = torch.nn.functional.mse_loss
    else:
        loss_fn = torch.nn.functional.cross_entropy
    
    model.eval()
    with torch.no_grad():
        data_iterator = tqdm(sensor_loader, desc="Predicting", unit="batch", position=0, leave=False)
        data_iterator.set_description(f"Predicting on sensor {sensor_loader.dataset.data_type} data")
        
        accuracies = np.array([])
        losses = np.array([])
        outputs = np.array([])
        fullnesses = np.array([])
        
        for (frame_data, target) in data_iterator:
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
            # print("features shape:", features.shape)
            # print("output shape (post squeeze, before feature_regressor):", output.shape)
            if feature_regressor is not None:
                output = output.flatten(1)
                output = feature_regressor(output)
                # output = torch.clamp(output, 0, 1)
                
            loss = loss_fn(output, fullness)
            losses = np.append(losses, loss.item())
            
            if not args.cont:
                output = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                _, target_fullness = torch.max(fullness, 1)
                correct = sum(predicted == target_fullness)
                accuracy = 100 * correct / args.batch_size
                accuracies = np.append(accuracies, accuracy.item())
                output = predicted
                fullness = target_fullness
            else:
                # Calculate accuracy of the model
                # Any prediction within x% of the true value is considered correct
                acceptable = 0.1
                accuracies = np.sum(np.abs(outputs - fullnesses) < acceptable) / len(outputs)
            
            outputs = np.append(outputs, output.cpu().numpy())
            fullnesses = np.append(fullnesses, fullness.cpu().numpy())
        
        if not args.cont:
            accuracies = accuracies.mean()
            
        if not args.no_wandb:
            wandb.log({f'Mean sensor prediction loss': losses.mean()})
            wandb.log({f'Mean sensor prediction accuracies': accuracies})
        
    
    return outputs, fullnesses, accuracies, losses

def plot_cont_predictions(outputs, fullnesses):
    """Plot the continuous predictions.
    """
    
    outputs = np.array(outputs)
    fullnesses = np.array(fullnesses)
    
    MSE = np.square(np.subtract(outputs, fullnesses)).mean()
    MAE = np.abs(np.subtract(outputs, fullnesses)).mean()
    RMSE = np.sqrt(MSE)
    PRMSE = np.sqrt(np.mean(np.sum(np.square(np.subtract(outputs, fullnesses)))) / np.sum(np.square(fullnesses)))
    
    # Calculate accuracy of the model
    # Any prediction within x% of the true value is considered correct
    acceptable = 0.1
    accuracy = np.sum(np.abs(outputs - fullnesses) < acceptable) / len(outputs)
    
    model_name = f'{args.model}_reg' if args.cont else f'{args.model}_cls'
    
    plt.figure(figsize=(10, 5))
    plt.plot(outputs, label='Predicted')
    plt.plot(fullnesses, label='True')
    
    # Plot acceptable range
    plt.fill_between(np.arange(len(outputs)), fullnesses-acceptable, fullnesses+acceptable, color='gray', alpha=0.5, label=f'threshold={acceptable*100:.0f}%')
    
    plt.legend()
    yticks = np.arange(0, 1.1, 0.1)
    plt.yticks(yticks)
    plt.grid()
    plt.xlabel('Frame')
    plt.ylabel('Fullness')
    plt.title(f'{model_name} Predicted vs True Fullness, MSE: {MSE:.3f}, MAE: {MAE:.3f}, RMSE: {RMSE:.3f}, PRMSE: {PRMSE:.3f}, accuracy: {accuracy*100:.1f}%')
    plt.tight_layout()
    
    if not args.no_wandb:
        title = f'Sensor predictions'
        wandb.log({title: wandb.Image(plt)})
    
    plt.close() 

def plot_confusion_matrix(outputs, fullness, num_classes):
    """Plot the confusion matrix for the model.
    """
    
    accuracy = np.sum(outputs == fullness) / len(outputs)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(outputs)):
        confusion_matrix[int(fullness[i]), int(outputs[i])] += 1
    
    labels = [f'{i*10}%' for i in range(num_classes)]
    
    plt.imshow(confusion_matrix, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{args.model}_cls Confusion Matrix, accuracy: {accuracy*100:.1f}%')
    # Set xticks and yticks with labels
    plt.xticks(ticks=np.arange(num_classes), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(num_classes), labels=labels)
    
    # Plot numbers on each square
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black' if confusion_matrix[i, j] > np.max(confusion_matrix)/2 else 'white')
    plt.tight_layout()
    
    if not args.no_wandb:
        title = f'Sensor predictions'
        wandb.log({title: wandb.Image(plt)})
    
    plt.close() 


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
        prediction = np.round(prediction*100)
        target = np.round(target*100)
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
            'data_subsample': args.data_subsample,
            'augment_probability': args.augment_probability,
            'use_sigmoid': args.use_sigmoid,
            'use_wce': args.use_wce,
            'optim': args.optim,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'pretrained': args.pretrained,
            'id': args.id,
            'greyscale': args.greyscale,
        })


def get_args():
    """Get the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train the CNN classifier model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for training')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the model to')
    parser.add_argument('--data_path', type=str, default='data/interim/chute_detection.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', type=bool, default=False, help='Include heatmaps in the training data')
    parser.add_argument('--inc_edges', type=bool, default=False, help='Include edges in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_wandb', action='store_true', help='Do not use Weights and Biases for logging')
    parser.add_argument('--model', type=str, default='vit', help='Model to use for training', choices=['cnn', 'convnextv2', 'vit', 'eva02', 'caformer'])
    parser.add_argument('--image_size', type=int, default=[1370, 204], help='Image size for the model (only for CNN)', nargs=2)
    parser.add_argument('--num_classes_straw', type=int, default=11, help='Number of classes for the straw classifier (11 = 10%, 21 = 5%)')
    parser.add_argument('--cont', action='store_true', help='Set model to predict a continuous value instead of a class')
    parser.add_argument('--data_subsample', type=float, default=1.0, help='Amount of the data to subsample for training (1.0 = 100%, 0.5 = 50%)')
    parser.add_argument('--augment_probability', type=float, default=0.5, help='Probability of augmenting the data')
    parser.add_argument('--use_sigmoid', action='store_true', help='Use a sigmoid activation function for the output')
    parser.add_argument('--use_wce', action='store_true', help='Use weighted cross-entropy loss')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer to use for training', choices=['adam', 'adamw', 'sgd', 'soap', 'adopt'])
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for the optimizer')
    parser.add_argument('--pretrained', action='store_true', help='Use a pretrained model')
    parser.add_argument('--id', type=str, default='', help='ID for the Weights and Biases run')
    parser.add_argument('--greyscale', action='store_true', help='Use greyscale images')
    parser.add_argument('--hpc', action='store_true', help='Run on the HPC')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # Is this how we want to do it?
    if args.hpc:
        path_to_data = f'/work3/davos/data/'
        if not os.path.exists(path_to_data):
            raise FileNotFoundError(f'Path to data not found: {path_to_data}')
        os.makedirs(f'{path_to_data}', exist_ok=True)
        args.data_path = path_to_data + args.data_path
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f'Data path not found: {args.data_path}')
        args.save_path = f'/work3/davos/models/'
        os.makedirs(args.save_path, exist_ok=True)
        if not os.path.exists(args.save_path):
            raise FileNotFoundError(f'Save path not found: {args.save_path}')
        sensor_path = f'/work3/davos/data/sensors.hdf5'
        if not os.path.exists(sensor_path):
            raise FileNotFoundError(f'Sensor data path not found: {sensor_path}')
        
    
    args.image_size = tuple(args.image_size)
    print(f'Using image size: {args.image_size}')
    
    match args.model:
        case 'cnn':
            image_size = args.image_size
        case 'convnextv2':
            image_size = (224, 224)
        case 'vit' | 'caformer':
            image_size = (384, 384)
        case 'eva02':
            image_size = (448, 448)
            
    image_size = args.image_size
    
    train_set = dl.Chute(data_path=args.data_path, data_type='train', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
                         random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
                         num_classes_straw=args.num_classes_straw, continuous=args.cont, subsample=args.data_subsample, augment_probability=0.5, greyscale=args.greyscale)
    # test_set = dl.Chute(data_path=args.data_path, data_type='test', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
    #                     random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
    #                     num_classes_straw=args.num_classes_straw, continuous=args.cont, subsample=args.data_subsample, augment_probability=0.0)
    
    mean, std = train_set.train_mean, train_set.train_std
    if args.inc_heatmap:
        mean_hm = train_set.train_hm_mean
        std_hm = train_set.train_hm_std
    
    statistics = (mean, std) if not args.inc_heatmap else (mean, std, mean_hm, std_hm)
    # test_set = dl.Chute(data_path=args.data_path, data_type='val', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
    #                     random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
    #                     num_classes_straw=args.num_classes_straw, continuous=args.cont)
    sensor_set = dl.Chute(data_path=sensor_path, data_type='test', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
                          random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, continuous=args.cont, subsample=1.0,
                          augment_probability=0, num_classes_straw=args.num_classes_straw, override_statistics=statistics, greyscale=args.greyscale)
    
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    sensor_loader = DataLoader(sensor_set, batch_size=args.batch_size, shuffle=False)
    
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
            model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=args.num_classes_straw, use_sigmoid=args.use_sigmoid)
        case 'convnextv2':
            model = timm.create_model('convnextv2_base.fcmae_ft_in222k_in1k_384', in_chans=input_channels, num_calsses=args.num_classes_straw, pretrained=args.pretrained)
        case 'convnext':
            model = timm.create_model('convnext_small.in12k_ft_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained)
        case 'vit':
            model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained, img_size=image_size)
        case 'eva02':
            model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained, img_size=image_size)
        case 'caformer':
                model = timm.create_model('caformer_m36.sail_in22k_ft_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained, img_size=image_size)
    
    if args.cont and args.model != 'cnn':
        features = model.forward_features(torch.randn(1, input_channels, image_size[0], image_size[1]))
        feature_size = torch.flatten(features, 1).shape[1]
        feature_regressor = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1, use_sigmoid=args.use_sigmoid)
    else:
        feature_regressor = None
    
    train_model(args, model, train_loader, sensor_loader, feature_regressor)
    
    # Load best model
    feature_regressor = None
    id = f'_{args.id}' if args.id != '' else ''
    if args.cont and args.model != 'cnn':
        in_feats = torch.randn(1, input_channels, image_size[0], image_size[1])
        in_feats = in_feats.cuda() if torch.cuda.is_available() else in_feats
        features = model.forward_features(in_feats)
        feature_size = torch.flatten(features, 1).shape[1]
        feature_regressor = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1)
        model.load_state_dict(torch.load(f'{args.save_path}/{args.model}_regressor/{args.model}_feature_extractor{id}_best.pth', weights_only=True))
        feature_regressor.load_state_dict(torch.load(f'{args.save_path}/{args.model}_regressor/{args.model}_regressor{id}_best.pth', weights_only=True))
        
    else:
        feature_regressor = None
        model.load_state_dict(torch.load(f'{args.save_path}/{args.model}_classifier/{args.model}_classifier{id}_best.pth', weights_only=True))

    outputs, fullnesses, accuracies, losses = predict_sensor_data(args, model, sensor_loader, feature_regressor)
    if args.cont:
        plot_cont_predictions(outputs, fullnesses)
    else:
        plot_confusion_matrix(outputs, fullnesses, num_classes=args.num_classes_straw)
        
    if not args.no_wandb:
        wandb.finish()
    
    # Delete the model after 60 seconds
    print(f'Deleting {args.model} models with id {id} in 60 seconds...')
    time.sleep(60)
    
    if args.cont and args.model != 'cnn':
        os.remove(f'{args.save_path}/{args.model}_regressor/{args.model}_feature_extractor{id}_best.pth')
        print(f'Deleted {args.model}_feature_extractor{id}_best.pth')
        os.remove(f'{args.save_path}/{args.model}_regressor/{args.model}_regressor{id}_best.pth')
        print(f'Deleted {args.model}_regressor{id}_best.pth')
    else:
        if args.cont:
            os.remove(f'{args.save_path}/{args.model}_regressor/{args.model}_regressor{id}_best.pth')
            print(f'Deleted {args.model}_regressor{id}_best.pth')
        else:
            os.remove(f'{args.save_path}/{args.model}_classifier/{args.model}_classifier{id}_best.pth')
            print(f'Deleted {args.model}_classifier{id}_best.pth')
    