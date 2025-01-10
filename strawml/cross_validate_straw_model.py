from __init__ import *

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import time
import timeit
import timm
import wandb
import os
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import data.dataloader as dl
import strawml.models.straw_classifier.cnn_classifier as cnn
import strawml.models.straw_classifier.feature_model as feature_model


def train_model(args, model: torch.nn.Module, train_loader: DataLoader, val_loader: torch.utils.data.DataLoader, feature_regressor: torch.nn.Module = None) -> None:
    """Train the CNN classifier model.
    """
    
    global OVERALL_BEST_ACCURACY
    global fold
    global LOG_DICT
    global BEST_FOLD
    
    if torch.cuda.is_available():
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
        best_accuracy = np.inf
    else:
        if args.use_wce:
            total = len(train_loader.dataset)
            class_counts = np.array(list(train_loader.dataset.class_counts.values()))
            # Using CW_i = N / C_i 
            ce_weights = torch.Tensor([total/class_counts]).flatten()
            ce_weights = ce_weights.cuda() if torch.cuda.is_available() else ce_weights
            # Using CW_i = N / (C_i * K)
            # ce_weights = torch.Tensor([total/(class_counts * train_set.num_classes_straw)])
        else:
            ce_weights = None

        loss_fn = torch.nn.functional.cross_entropy
        best_accuracy = -1.0
    
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []
    
    plot_examples_every = 0.5 # Plot an example every X% of the length of the training data
    
    for epoch in range(args.epochs):
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        train_iterator.set_description(f'Fold {fold+1}/{args.folds} Training Epoch {epoch+1}/{args.epochs}')
        model.train()
        if feature_regressor is not None:
            feature_regressor.train()
        train_accuracies = []
        train_losses = []
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
            
            # print("frame_data shape:", frame_data.shape)
            
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
            
            if args.use_wce and not args.cont:
                loss = loss_fn(output, fullness, weight=ce_weights)
            else:
                loss = loss_fn(output, fullness)
            train_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            if not args.only_head and feature_regressor is not None:
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
                train_accuracies.append(accuracy.item())
        
                train_iterator.set_postfix(loss=loss.item(), accuracy=sum(train_accuracies)/len(train_accuracies))

            if not args.no_wandb:
                # Save an example every 10% of the length of the training data
                training_info = {'data_type': 'train', 'current_iteration': current_iteration, 'epoch': epoch+1, 'fold': fold+1}
                divisor = int(len(train_loader) * plot_examples_every)
                divisor = 1 if divisor == 0 else divisor
                if current_iteration % divisor == 0:
                    if args.cont:
                        plot_example(training_info, frame_data, output, fullness)
                    else:
                        plot_example(training_info, frame_data, predicted, target_fullness)
            
        mean_train_loss = np.mean(train_losses)
        epoch_train_losses.append(mean_train_loss)
        if not args.cont:
            mean_train_accuracy = np.mean(train_accuracies)
            epoch_train_accuracies.append(mean_train_accuracy)
        
        if not args.no_wandb:
            if not args.cont:
                LOG_DICT[f'f{fold+1}_train_loss'] = mean_train_loss
                LOG_DICT[f'f{fold+1}_train_accuracy'] = mean_train_accuracy
                # wandb.log(data={f'f{fold+1}_train_loss': mean_train_loss, f'f{fold+1}_train_accuracy': mean_train_accuracy})
            else:
                LOG_DICT[f'f{fold+1}_train_loss'] = mean_train_loss
                # wandb.log(data={f'f{fold+1}_train_loss': mean_train_loss})
            
            
        if args.cont:
            print(f'Epoch: {epoch+1}, Training Loss: {sum(train_losses)/len(train_losses):.6f}, Last predictions -- Fullness: {torch.mean(fullness).item():.2f}, Prediction: {torch.mean(output).item():.2f}')
        else:
            print(f'Epoch: {epoch+1}, Training Accuracy: {sum(train_accuracies)/len(train_accuracies):.2f}%')
        
        model.eval()
        if feature_regressor is not None:
            feature_regressor.eval()
        
        with torch.no_grad():
            correct = 0
            total = 0
            val_iterator = tqdm(val_loader, unit="batch", position=0, leave=False)
            val_iterator.set_description(f'Fold {fold+1}/{args.folds} Validating Epoch {epoch+1}/{args.epochs}')
            
            
            val_losses = []
            val_accuracies = []
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
                val_losses.append(val_loss.item())
                
                if not args.cont:
                    output = torch.nn.functional.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    total += args.batch_size
                    _, target_fullness = torch.max(fullness, 1)
                    correct += sum(predicted == target_fullness)
                    accuracy = 100 * correct / args.batch_size
                    val_accuracies.append(accuracy.item())

                if not args.no_wandb:
                    # Save an example every 10% of the length of the training data
                    val_info = {'data_type': 'val', 'current_iteration': current_iteration, 'epoch': epoch+1, 'fold': fold+1}
                    divisor = int(len(val_loader) * plot_examples_every)
                    divisor = 1 if divisor == 0 else divisor
                    if current_iteration % divisor == 0:
                        if args.cont:
                            plot_example(val_info, frame_data, output, fullness)
                        else:
                            plot_example(val_info, frame_data, predicted, target_fullness)
                
                
            average_time = np.mean(batch_times)
            if not args.no_wandb:
                LOG_DICT[f'f{fold+1}_inference_time'] = average_time
            
            # Save the best model
            if args.cont:
                print(f'Epoch: {epoch+1}, Average Inference Time: {average_time:.6f} seconds, Total Inference Time: {np.sum(batch_times):.6f} seconds. (Batch Size: {args.batch_size}) Validation Loss: {np.mean(val_losses):.6f}, Last predictions -- Fullness: {torch.mean(fullness).item():.2f}, Prediction: {torch.mean(output).item():.2f}')
                if np.mean(val_losses) < best_accuracy:
                    best_accuracy = np.mean(val_losses)
                    print(f'New best loss: {best_accuracy}')
                    
                    model_folder = f'{args.save_path}{args.model}_regressor/'
                    os.makedirs(model_folder, exist_ok=True)
                    id = f'_{args.id}' if args.id != '' else ''
                    if args.model != 'cnn':
                        model_name = f'{args.model}_feature_extractor{id}'
                        model_save_path = model_folder + model_name + '_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                        regressor_name = f'{args.model}_regressor{id}'
                        regressor_save_path = model_folder + regressor_name + '_best.pth'
                        torch.save(feature_regressor.state_dict(), regressor_save_path)
                    else:
                        model_name = f'{args.model}_regressor{id}'
                        model_save_path = model_folder + model_name + '_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                    # if not args.no_wandb:
                    #     wandb.save(model_save_path)
                    #     wandb.save(regressor_save_path)
                    
                            
                    if not args.no_wandb:
                        LOG_DICT[f'f{fold+1}_best_val_loss'] = best_accuracy
                        # wandb.log(data={f'f{fold+1}_best_val_loss': best_accuracy})
            else:
                correct = correct.detach().cpu()
                accuracy = 100 * correct /total
                print(f'Epoch: {epoch+1}, Validation Accuracy: {accuracy:.2f}%. Average Inference Time: {average_time:.6f} seconds, Total Inference Time: {sum(batch_times):.6f} seconds. (Batch Size: {args.batch_size})')

                if not args.no_wandb:
                    LOG_DICT[f'f{fold+1}_val_accuracy'] = accuracy
                    # wandb.log(data={f'f{fold+1}_val_accuracy': accuracy,})
                
                if accuracy > best_accuracy:
                    print(f'New best accuracy: {accuracy:.2f}%')
                    best_accuracy = accuracy
                    model_folder = f'{args.save_path}{args.model}_classifier/'
                    os.makedirs(model_folder, exist_ok=True)
                    id = f'_{args.id}' if args.id != '' else ''
                    model_name = f'{args.model}_classifier{id}'
                    model_save_path = model_folder + model_name + '_best.pth'
                    torch.save(model.state_dict(), model_save_path)
                    # if not args.no_wandb:
                    #     wandb.save(model_save_path)
            
            if args.cont:
                if np.mean(val_losses) < OVERALL_BEST_ACCURACY:
                    print('----------------------------------------------------')
                    print(f'New overall best loss: {best_accuracy} from {OVERALL_BEST_ACCURACY}')
                    BEST_FOLD = fold +1
                    OVERALL_BEST_ACCURACY = np.mean(val_losses)
                    print(f'New overall best loss: {OVERALL_BEST_ACCURACY:.6f}')
                    model_folder = f'{args.save_path}{args.model}_regressor/'
                    os.makedirs(model_folder, exist_ok=True)
                    id = f'_{args.id}' if args.id != '' else ''
                    if args.model != 'cnn':
                        model_name = f'{args.model}_feature_extractor{id}'
                        model_save_path = model_folder + model_name + '_overall_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                        regressor_name = f'{args.model}_regressor{id}'
                        regressor_save_path = model_folder + regressor_name + '_overall_best.pth'
                        torch.save(feature_regressor.state_dict(), regressor_save_path)
                    else:
                        model_name = f'{args.model}_regressor{id}'
                        model_save_path = model_folder + model_name + '_overall_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                    print(f'Saving {model_save_path} from fold {fold+1} as the overall best model')
                    print('----------------------------------------------------')
                    if not args.no_wandb:
                        wandb.save(model_save_path)
                        wandb.save(regressor_save_path)
            else:
                if best_accuracy > OVERALL_BEST_ACCURACY:
                    print('----------------------------------------------------')
                    print(f'New overall best accuracy: {best_accuracy}% up from {OVERALL_BEST_ACCURACY}%')
                    OVERALL_BEST_ACCURACY = best_accuracy
                    BEST_FOLD = fold + 1
                    model_folder = f'{args.save_path}{args.model}_classifier/'
                    os.makedirs(model_folder, exist_ok=True)
                    id = f'_{args.id}' if args.id != '' else ''
                    model_name = f'{args.model}_classifier{id}'
                    model_save_path = model_folder + model_name + '_overall_best.pth'
                    print(f'Saving {model_save_path} from fold {fold+1} as the overall best model')
                    print('----------------------------------------------------')
                    torch.save(model.state_dict(), model_save_path)
                    if not args.no_wandb:
                        wandb.save(model_save_path)
                
            if not args.no_wandb:
                LOG_DICT['custom_step'] = epoch+1
                LOG_DICT[f'f{fold+1}_val_loss'] = np.mean(val_losses)
                LOG_DICT[f'f{fold+1}_epoch'] = epoch+1
                
                # wandb.log(data={f'f{fold+1}_val_loss': np.mean(val_losses), 
                #               f'f{fold+1}_epoch': epoch+1})
                
                wandb.log(LOG_DICT)
            
            epoch_val_losses.append(np.mean(val_losses))
            if not args.cont:
                epoch_val_accuracies.append(accuracy)
    
    return epoch_train_losses, epoch_val_losses, epoch_train_accuracies, epoch_val_accuracies, best_accuracy

def predict_sensor_data(args, model: torch.nn.Module, sensor_loader: DataLoader, feature_regressor: torch.nn.Module = None, in_fold=True):
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
            if in_fold:
                title = f'f{fold+1}'
            else:
                title = 'Overall'
            wandb.log({f'{title} mean sensor prediction loss': losses.mean()})
            wandb.log({f'{title} mean sensor prediction accuracies': accuracies})
        
    
    return outputs, fullnesses, accuracies, losses
    

def plot_cont_predictions(outputs, fullnesses, in_fold=True):
    """Plot the continuous predictions.
    """
    
    global BEST_SENSOR_FOLD
    
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
    if not in_fold: model_name = f'{model_name}_f{BEST_SENSOR_FOLD}' 
    
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
        if in_fold:
            title = f'f{fold+1} sensor predictions'
        else:
            title = 'Overall sensor predictions'
        wandb.log({title: wandb.Image(plt)})
    
    plt.close() 
    
def plot_confusion_matrix(outputs, fullness, num_classes, in_fold=True):
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
        if in_fold:
            title = f'f{fold+1} sensor predictions'
        else:
            title = 'Overall sensor predictions'
        wandb.log({title: wandb.Image(plt)})
    
    plt.close() 

def plot_example(info, frame_data, prediction, target):
    """Plot an example frame with the prediction.
    """
    if len(frame_data.shape) == 4:
        frame_data = frame_data[0]
    if len(prediction) > 1:
        prediction = prediction[0]
    if len(target) > 1:
        target = target[0]
    if frame_data.shape[0] > 3:
        frame_data = frame_data[:3]
    
    if args.greyscale:
        frame_data = frame_data[0]
        frame_data = frame_data.unsqueeze(0)
    
    frame_data = frame_data.permute(1, 2, 0)
    frame_data = frame_data.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    means = train_set.train_mean
    stds = train_set.train_std
    frame_data = frame_data * stds + means
    frame_data = np.clip(frame_data, 0, 255)
    frame_data = frame_data.astype(np.uint8)
    
    if args.cont:
        prediction = int(prediction*100)
        target = int(target*100)
    else:
        increment = increment = 100 / (train_set.num_classes_straw - 1)
        prediction = prediction * increment
        target = target * increment
    
    if args.greyscale:
        plt.imshow(frame_data, cmap="gray")
    else:
        plt.imshow(frame_data)
    plt.title(f'{info["data_type"]} Epoch: {info["epoch"]} it: {info["current_iteration"]} Prediction: {prediction} Target: {target}')
    
    # plt.show()
    
    if not args.no_wandb:
        wandb.log({f'f{info["fold"]}_{info["data_type"]}_example_frame': wandb.Image(plt)})
    
        
    plt.close()


def initialize_wandb(args: argparse.Namespace) -> None:
    """Initialize the Weights and Biases logging.
    """
    wandb.login()
    wandb.init(
        project='cv-testrun',
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
            'folds': args.folds,
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
            'only_head': args.only_head,
            'num_hidden_layers': args.num_hidden_layers,
            'num_neurons': args.num_neurons,
            'balanced_dataset': args.balanced_dataset
        })
    
    global LOG_DICT
    LOG_DICT = {}
    wandb.define_metric('custom_step')
    for fold in range(args.folds):
        LOG_DICT['custom_step'] = 0
        LOG_DICT[f'f{fold+1}_train_loss'] = None
        LOG_DICT[f'f{fold+1}_val_loss'] = None
        LOG_DICT[f'f{fold+1}_train_accuracy'] = None
        LOG_DICT[f'f{fold+1}_val_accuracy'] = None
        LOG_DICT[f'f{fold+1}_best_val_loss'] = None
        LOG_DICT[f'f{fold+1}_best_val_accuracy'] = None
        LOG_DICT[f'f{fold+1}_epoch'] = None
        LOG_DICT[f'f{fold+1}_inference_time'] = None
        # LOG_DICT[f'f{fold+1}_train_example_frame'] = None
        # LOG_DICT[f'f{fold+1}_val_example_frame'] = None
        
        wandb.define_metric(f'f{fold+1}_train_loss', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_val_loss', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_train_accuracy', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_val_accuracy', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_best_val_loss', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_best_val_accuracy', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_epoch', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_inference_time', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_train_data_type_example_frame', step_metric='custom_step')
        wandb.define_metric(f'f{fold+1}_val_data_type_example_frame', step_metric='custom_step')

def get_args():
    """Get the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train the CNN classifier model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for training')
    parser.add_argument('--model_path', type=str, default='models/pretrained/convnextv2_atto_1k_224_ema.pth', help='Path to load the model from')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the model to')
    parser.add_argument('--data_path', type=str, default='data/interim/chute_detection.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', action='store_true', help='Include heatmaps in the training data')
    parser.add_argument('--inc_edges', action='store_true', help='Include edges in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_wandb', action='store_true', help='Do not use Weights and Biases for logging')
    parser.add_argument('--model', type=str, default='vit', help='Model to use for training', choices=['cnn', 'convnext', 'convnextv2', 'vit', 'eva02', 'caformer'])
    parser.add_argument('--image_size', type=int, default=[1370, 204], help='Image size for the model (only for CNN)', nargs=2)
    parser.add_argument('--num_classes_straw', type=int, default=11, help='Number of classes for the straw classifier (11 = 10%, 21 = 5%)')
    parser.add_argument('--cont', action='store_true', help='Set model to predict a continuous value instead of a class')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--data_subsample', type=float, default=1, help='Amount of the data to subsample for training (1.0 = 100%, 0.5 = 50%)')
    parser.add_argument('--augment_probability', type=float, default=0.5, help='Probability of augmenting the data')
    parser.add_argument('--use_sigmoid', action='store_true', help='Use a sigmoid activation function for the output')
    parser.add_argument('--use_wce', action='store_true', help='Use weighted cross-entropy loss')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer to use for training', choices=['adam', 'adamw', 'sgd', 'soap', 'adopt'])
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for the optimizer')
    parser.add_argument('--pretrained', action='store_true', help='Use a pretrained model')
    parser.add_argument('--id', type=str, default='', help='ID for the run')
    parser.add_argument('--greyscale', action='store_true', help='Use greyscale images')
    parser.add_argument('--hpc', action='store_true', help='Use the HPC')
    parser.add_argument('--only_head', action='store_true', help='Only train the head of the model')
    parser.add_argument('--num_hidden_layers', type=int, default=0, help='Number of hidden layers for the regressor. Default: 0 (in->out)')
    parser.add_argument('--num_neurons', type=int, default=512, help='Number of neurons for the regressor')
    parser.add_argument('--balanced_dataset', action='store_true', help='Balance the dataset setting the maximum number of samples in each class to a max of 400')
    parser.add_argument('--is_sweep', action='store_true', help='Is this a sweep run?')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    if args.cont:
        print(f'Training regression model with {args.model}')
    else:
        print(f'Training classification model with {args.model}')
    
    if args.inc_edges:
        print('Including edges in the training data')
    if args.inc_heatmap:
        print('Including heatmaps in the training data')
    
    # Is this how we want to do it?
    ## NOTE SET NAME OF WORK3 USER
    # pers_id = "s194247"
    pers_id = "davos"
    if args.hpc:
        path_to_data = f'/work3/{pers_id}/data/'
        if not os.path.exists(path_to_data):
            raise FileNotFoundError(f'Path to data not found: {path_to_data}')
        os.makedirs(f'{path_to_data}', exist_ok=True)
        args.data_path = path_to_data + args.data_path
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f'Data path not found: {args.data_path}')
        args.save_path = f'/work3/{pers_id}/models/'
        os.makedirs(args.save_path, exist_ok=True)
        if not os.path.exists(args.save_path):
            raise FileNotFoundError(f'Save path not found: {args.save_path}')
        sensor_path = f'/work3/{pers_id}/data/sensors.hdf5'
        if not os.path.exists(sensor_path):
            raise FileNotFoundError(f'Sensor data path not found: {sensor_path}')
    
    if torch.cuda.is_available(): print('Using GPU')
    
    args.image_size = tuple(args.image_size)
    
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
    
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    train_set = dl.Chute(data_path=args.data_path, data_type='train', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
                         random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
                         num_classes_straw=args.num_classes_straw, continuous=args.cont, subsample=args.data_subsample, 
                         augment_probability=args.augment_probability, greyscale=args.greyscale, balance_dataset=args.balanced_dataset)
    
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
    
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accuracies = []
    fold_val_accuracies = []
    best_accuracies = []
    global OVERALL_BEST_ACCURACY
    OVERALL_BEST_ACCURACY = -1.0 if not args.cont else torch.inf
    global LOG_DICT
    global BEST_FOLD
    global BEST_SENSOR_FOLD
    
    BEST_SENSOR_ACCURACY = -1.0
    
    # WANDB
    if not args.no_wandb:
        initialize_wandb(args)
        if args.is_sweep:
            wandb.config.update(args, allow_val_change=True)
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
        if not args.no_wandb:
            LOG_DICT['custom_step'] = 0
            LOG_DICT[f'f{fold}_train_loss'] = None
            LOG_DICT[f'f{fold}_val_loss'] = None
            LOG_DICT[f'f{fold}_train_accuracy'] = None
            LOG_DICT[f'f{fold}_val_accuracy'] = None
            LOG_DICT[f'f{fold}_best_val_loss'] = None
            LOG_DICT[f'f{fold}_best_val_accuracy'] = None
            LOG_DICT[f'f{fold}_epoch'] = None
            LOG_DICT[f'f{fold}_inference_time'] = None
            # LOG_DICT[f'f{fold}_train_data_type_example_frame'] = None
            # LOG_DICT[f'f{fold}_val_data_type_example_frame'] = None
        
        print(f'Training fold {fold+1}/{args.folds}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_idx))
        
        input_channels = 3 if not args.greyscale else 1
        if args.inc_heatmap: input_channels += 3
        if args.inc_edges: input_channels += 1
        
        if args.cont:
            args.num_classes_straw = 1
        
        match args.model:
            case 'cnn':
                model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=args.num_classes_straw, use_sigmoid=args.use_sigmoid)
            case 'convnextv2':
                model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained)
            case 'convnext':
                model = timm.create_model('convnext_small.in12k_ft_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained)
            case 'vit':
                model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained, img_size=image_size)
            case 'eva02':
                model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained, img_size=image_size)
            case 'caformer':
                model = timm.create_model('caformer_m36.sail_in22k_ft_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=args.pretrained, img_size=image_size)
        
        if args.cont and args.model != 'cnn':
            in_feats = torch.randn(1, input_channels, image_size[0], image_size[1])
            # in_feats = in_feats.cuda() if torch.cuda.is_available() else in_feats
            features = model.forward_features(in_feats)
            feature_size = torch.flatten(features, 1).shape[1]
            feature_regressor = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1, use_sigmoid=args.use_sigmoid, num_hidden_layers=args.num_hidden_layers, num_neurons=args.num_neurons)
        else:
            feature_regressor = None
        
        train_loss, val_loss, train_accuracy, val_accuracy, best_accuracy = train_model(args, model, train_loader, val_loader, feature_regressor)

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_accuracies.append(train_accuracy)
        fold_val_accuracies.append(val_accuracy)
        best_accuracies.append(best_accuracy)
        
        id = f'_{args.id}' if args.id != '' else ''
        if args.cont and args.model != 'cnn':
            model.load_state_dict(torch.load(f'{args.save_path}/{args.model}_regressor/{args.model}_feature_extractor{id}_best.pth', weights_only=True))
            feature_regressor.load_state_dict(torch.load(f'{args.save_path}/{args.model}_regressor/{args.model}_regressor{id}_best.pth', weights_only=True))
        else:
            feature_regressor = None
            model.load_state_dict(torch.load(f'{args.save_path}/{args.model}_classifier/{args.model}_classifier{id}_best.pth', weights_only=True))
        
        sensor_loader = DataLoader(sensor_set, batch_size=args.batch_size)
        outputs, fullnesses, accuracies, losses = predict_sensor_data(args, model, sensor_loader, feature_regressor)
        
        acceptable = 0.1
        sensor_accuracy = np.sum(np.abs(outputs - fullnesses) < acceptable) / len(outputs)
        if sensor_accuracy > BEST_SENSOR_ACCURACY:
            BEST_SENSOR_ACCURACY = sensor_accuracy
            BEST_SENSOR_FOLD = fold + 1
            model_folder = f'{args.save_path}/{args.model}' + ('_regressor/' if args.cont else '_classifier/')
            id = f'_{args.id}' if args.id != '' else ''
            model_suffix = 'feature_extractor' if args.cont else 'classifier'
            model_save_path = f'{model_folder}{args.model}_{model_suffix}{id}'
            torch.save(model.state_dict(), f'{model_save_path}_best_sensor.pth')
            if feature_regressor is not None:
                torch.save(feature_regressor.state_dict(), f'{model_folder}{args.model}_regressor{id}_best_sensor.pth')

            if not args.no_wandb:
                wandb.save(f'{model_save_path}_best_sensor.pth')
                if feature_regressor is not None:
                    wandb.save(f'{model_folder}{args.model}_regressor{id}_best_sensor.pth')
            
        if args.cont:
            plot_cont_predictions(outputs, fullnesses, in_fold=True)
        else:
            plot_confusion_matrix(outputs, fullnesses, num_classes=args.num_classes_straw, in_fold=True)
        
        # if args.cont:
        #     if np.mean(val_loss) < OVERALL_BEST_ACCURACY:
        #         OVERALL_BEST_ACCURACY = np.mean(val_loss)
        # else:
        #     if best_accuracy > OVERALL_BEST_ACCURACY:
        #         OVERALL_BEST_ACCURACY = best_accuracy
    
    # mean_best_val_loss = np.mean(fold_val_losses)
    # std_best_val_loss = np.std(fold_val_losses)
    
    # if not args.cont:
    #     mean_best_val_accuracy = np.mean(best_accuracies)
    #     std_best_val_accuracy = np.std(best_accuracies)
    #     print(f'Mean Best Validation Accuracy: {mean_best_val_accuracy:.2f}% +/- {std_best_val_accuracy:.2f}%')
    
    # print(f'Mean Best Validation Loss: {mean_best_val_loss:.4f} +/- {std_best_val_loss:.4f}')
    
    # if args.cont and args.model != 'cnn':
    #     id = f'_{args.id}' if args.id != '' else ''
    #     model.load_state_dict(torch.load(f'{args.save_path}/{args.model}_regressor/{args.model}_feature_extractor{id}_overall_best.pth', weights_only=True))
    #     feature_regressor.load_state_dict(torch.load(f'{args.save_path}/{args.model}_regressor/{args.model}_regressor{id}_overall_best.pth', weights_only=True))
        
    # else:
    #     feature_regressor = None
    #     model.load_state_dict(torch.load(f'{args.save_path}/{args.model}_classifier/{args.model}_classifier{id}_overall_best.pth', weights_only=True))
    
    model_folder = f'{args.save_path}/{args.model}' + ('_regressor/' if args.cont else '_classifier/')
    id = f'_{args.id}' if args.id != '' else ''
    model_suffix = 'feature_extractor' if args.cont else 'classifier'
    model_save_path = f'{model_folder}{args.model}_{model_suffix}{id}'
    model.load_state_dict(torch.load(f'{model_save_path}_best_sensor.pth', weights_only=True))
    if feature_regressor is not None:
        feature_regressor.load_state_dict(torch.load(f'{model_folder}{args.model}_regressor{id}_best_sensor.pth', weights_only=True))
    
    sensor_loader = DataLoader(sensor_set, batch_size=args.batch_size)
    outputs, fullnesses, accuracies, losses = predict_sensor_data(args, model, sensor_loader, feature_regressor, in_fold=False)
    if args.cont:
        plot_cont_predictions(outputs, fullnesses, in_fold=False)
    else:
        plot_confusion_matrix(outputs, fullnesses, num_classes=args.num_classes_straw, in_fold=False)

    
    if not args.no_wandb:
        wandb.log({'best_fold': BEST_FOLD})
        wandb.log({'overall_best_acc_or_loss': OVERALL_BEST_ACCURACY})
        wandb.log({'best_sensor_fold': BEST_SENSOR_FOLD})
        wandb.log({'best_sensor_accuracy': BEST_SENSOR_ACCURACY})
        # if not args.cont:
        #     wandb.log({'mean_best_val_accuracy': mean_best_val_accuracy, 'std_best_val_accuracy': std_best_val_accuracy})
    
    model_folder = f'{args.model}_classifier/' if not args.cont else f'{args.model}_regressor/'
    for i in range(args.folds):
        x = np.arange(1, args.epochs+1, step=1)
        plt.plot(x, fold_train_losses[i], c='b', linestyle='--', alpha=0.2)
        plt.plot(x, fold_val_losses[i], c='r', alpha=0.2)
        
    plt.plot(x, np.mean(fold_train_losses, axis=0), c='b', label='Train Loss', linestyle='--')
    plt.plot(x, np.mean(fold_val_losses, axis=0), c='r', label='Val Loss')
    plt.ylim(0, min(7, np.max(fold_val_losses), np.max(fold_val_losses))+0.01)
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs(f'reports/figures/model_results/{model_folder}', exist_ok=True)
    plt.savefig(f'reports/figures/model_results/{model_folder}/loss_combined.png')
    
    if not args.no_wandb:
        wandb.log({'loss_combined': wandb.Image(plt)})
    
    plt.close()
    
    if not args.cont:
        for i in range(args.folds):
            x = np.arange(1, args.epochs+1, step=1)
            plt.plot(x, fold_train_accuracies[i], c='b', linestyle='--', alpha=0.2)
            plt.plot(x, fold_val_accuracies[i], c='r', alpha=0.2)
        
        plt.plot(x, np.mean(fold_train_accuracies, axis=0), c='b', label='Train Accuracy', linestyle='--')
        plt.plot(x, np.mean(fold_val_accuracies, axis=0), c='r', label='Val Accuracy')
        plt.ylim(0, 100)
        plt.title('Accuracy per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        os.makedirs(f'reports/figures/model_results/{model_folder}', exist_ok=True)
        plt.savefig(f'reports/figures/model_results/{model_folder}/accuracy_combined.png')
    
        if not args.no_wandb:
            wandb.log({'accuracy_combined': wandb.Image(plt)})
        
        plt.close()
    
    if not args.no_wandb: wandb.finish()
    
    # Delete the models to free up memory
    print(f'Deleting {args.model} models with id: {id} in 60 seconds...')
    time.sleep(60)
    
    os.remove(f'{model_save_path}_best_sensor.pth')
    print(f'Deleted {model_save_path}_best_sensor.pth')
    os.remove(f'{args.save_path}/{model_folder}{args.model}_{model_suffix}{id}_best.pth')
    print(f'Deleted {args.save_path}/{model_folder}{args.model}_{model_suffix}{id}_best.pth')
    os.remove(f'{args.save_path}/{model_folder}{args.model}_{model_suffix}{id}_overall_best.pth')
    print(f'Deleted {args.save_path}/{model_folder}{args.model}_{model_suffix}{id}_overall_best.pth')
    if feature_regressor is not None:
        os.remove(f'{args.save_path}/{model_folder}{args.model}_regressor{id}_best_sensor.pth')
        print(f'Deleted {args.save_path}/{model_folder}{args.model}_regressor{id}_best_sensor.pth')
        os.remove(f'{args.save_path}/{model_folder}{args.model}_regressor{id}_best.pth')
        print(f'Deleted {args.save_path}/{model_folder}{args.model}_regressor{id}_best.pth')
        os.remove(f'{args.save_path}/{model_folder}{args.model}_regressor{id}_overall_best.pth')
        print(f'Deleted {args.save_path}/{model_folder}{args.model}_regressor{id}_overall_best.pth')
    
        