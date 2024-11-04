from __init__ import *

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
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
    
    if torch.cuda.is_available():
        model = model.cuda()
        if feature_regressor is not None:
            feature_regressor = feature_regressor.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.cont:
        loss_fn = torch.nn.functional.mse_loss
        best_accuracy = np.inf
    else:
        loss_fn = torch.nn.functional.cross_entropy
        best_accuracy = -1.0
    
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []
    
    for epoch in range(args.epochs):
        train_iterator = tqdm(train_loader, unit="batch", position=0, leave=False)
        train_iterator.set_description(f'Training Epoch {epoch+1}/{args.epochs}')
        model.train()
        if feature_regressor is not None:
            feature_regressor.train()
        train_accuracies = []
        train_losses = []
        for (frame_data, target) in train_iterator:
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
            if args.cont: output = output.squeeze()
            if feature_regressor is not None:
                output = feature_regressor(output)
                output = output.squeeze()
            
            loss = loss_fn(output, fullness)
            train_losses.append(loss.item())
            
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
                train_accuracies.append(accuracy.item())
        
                train_iterator.set_postfix(loss=loss.item(), accuracy=sum(train_accuracies)/len(train_accuracies))

        mean_train_loss = np.mean(train_losses)
        mean_train_accuracy = np.mean(train_accuracies)
        epoch_train_losses.append(mean_train_loss)
        epoch_train_accuracies.append(mean_train_accuracy)
        
        if not args.no_wandb:
            if not args.cont:
                wandb.log(step=epoch+1, data={f'f{fold+1}_train_loss': mean_train_loss, f'f{fold+1}_train_accuracy': mean_train_accuracy})
            else:
                wandb.log(step=epoch+1, data={f'f{fold+1}_train_loss': mean_train_loss})
            
            
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
            val_iterator.set_description(f'Validating Epoch {epoch+1}/{args.epochs}')
            
            # Time the inference time
            start_time = timeit.default_timer()
            
            val_losses = []
            val_accuracies = []
            for (frame_data, target) in val_iterator:
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
            
            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            average_time = elapsed_time / len(val_loader)
            
            # Save the best model
            if args.cont:
                print(f'Epoch: {epoch+1}, Average Inference Time: {average_time:.2f} seconds, Total Inference Time: {elapsed_time:.2f} seconds. (Batch Size: {args.batch_size}) Validation Loss: {np.mean(val_losses):.6f}, Last predictions -- Fullness: {torch.mean(fullness).item():.2f}, Prediction: {torch.mean(output).item():.2f}')
                if np.mean(val_losses) < best_accuracy:
                    best_accuracy = np.mean(val_losses)
                    print(f'New best loss: {best_accuracy}')
                    # model_folder = f'{args.save_path}{args.model}_regressor/'
                    # os.makedirs(model_folder, exist_ok=True)
                    # if args.model != 'cnn':
                    #     model_name = args.model + f'f{fold+1}' + '_feature_extractor'
                    #     model_save_path = model_folder + model_name + '_best.pth'
                    #     regressor_name = args.model  + f'f{fold+1}' +  '_regressor'
                    #     regressor_save_path = model_folder + regressor_name + '_best.pth'
                    #     torch.save(model.state_dict(), model_save_path)
                    #     torch.save(feature_regressor.state_dict(), regressor_save_path)
                    #     if not args.no_wandb:
                    #         wandb.save(model_save_path)
                    #         wandb.save(regressor_save_path)
                    # else:
                    #     model_name = f'{args.model}_f{fold+1}_regressor'
                    #     model_save_path = model_folder + model_name + '_best.pth'
                    #     torch.save(model.state_dict(), model_save_path)
                    #     if not args.no_wandb:
                    #         wandb.save(model_save_path)
                            
                    if not args.no_wandb:
                        wandb.log(step=epoch+1, data={f'f{fold+1}_best_val_loss': best_accuracy})
            else:
                correct = correct.detach().cpu()
                accuracy = 100 * correct /total
                print(f'Epoch: {epoch+1}, Validation Accuracy: {accuracy:.2f}%. Average Inference Time: {average_time:.2f} seconds, Total Inference Time: {elapsed_time:.2f} seconds. (Batch Size: {args.batch_size})')

                if not args.no_wandb:
                    wandb.log(step=epoch+1, 
                              data={f'f{fold+1}_val_accuracy': accuracy,})
                
                if accuracy > best_accuracy:
                    print(f'New best accuracy: {accuracy:.2f}%')
                    best_accuracy = accuracy
                    # model_folder = f'{args.save_path}{args.model}_classifier/'
                    # os.makedirs(model_folder, exist_ok=True)
                    # model_name = f'{args.model}_f{fold+1}_classifier'
                    # model_save_path = model_folder + model_name + '_best.pth'
                    # torch.save(model.state_dict(), model_save_path)
                    # if not args.no_wandb:
                    #     wandb.log(step=epoch+1, data={f'f{fold+1}_best_val_accuracy': best_accuracy})
                    #     wandb.save(model_save_path)
            
            if args.cont:
                if np.mean(val_losses) < OVERALL_BEST_ACCURACY:
                    OVERALL_BEST_ACCURACY = np.mean(val_losses)
                    print(f'New overall best loss: {OVERALL_BEST_ACCURACY:.6f}')
                    model_folder = f'{args.save_path}{args.model}_regressor/'
                    os.makedirs(model_folder, exist_ok=True)
                    if args.model != 'cnn':
                        model_name = f'{args.model}_feature_extractor'
                        model_save_path = model_folder + model_name + '_overall_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                        regressor_name = f'{args.model}_regressor'
                        regressor_save_path = model_folder + regressor_name + '_overall_best.pth'
                        torch.save(feature_regressor.state_dict(), regressor_save_path)
                    else:
                        model_name = f'{args.model}_regressor'
                        model_save_path = model_folder + model_name + '_overall_best.pth'
                        torch.save(model.state_dict(), model_save_path)
                    if not args.no_wandb:
                        wandb.save(model_save_path)
                        wandb.save(regressor_save_path)
            else:
                if best_accuracy > OVERALL_BEST_ACCURACY:
                    OVERALL_BEST_ACCURACY = best_accuracy
                    print(f'New overall best accuracy: {OVERALL_BEST_ACCURACY}%')
                    model_folder = f'{args.save_path}{args.model}_classifier/'
                    os.makedirs(model_folder, exist_ok=True)
                    model_name = f'{args.model}_classifier'
                    model_save_path = model_folder + model_name + '_overall_best.pth'
                    torch.save(model.state_dict(), model_save_path)
                    if not args.no_wandb:
                        wandb.save(model_save_path)
                
            if not args.no_wandb:
                wandb.log(step=epoch+1, 
                        data={f'f{fold+1}_val_loss': np.mean(val_losses), 
                              'epoch': epoch+1})
            
            epoch_val_losses.append(np.mean(val_losses))
            if not args.cont:
                epoch_val_accuracies.append(accuracy)
    
    return epoch_train_losses, epoch_val_losses, epoch_train_accuracies, epoch_val_accuracies, best_accuracy
    
    

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
            'data_subsample': args.data_subsample
        })


def get_args():
    """Get the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train the CNN classifier model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for training')
    parser.add_argument('--model_path', type=str, default='models/pretrained/convnextv2_atto_1k_224_ema.pth', help='Path to load the model from')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the model to')
    parser.add_argument('--data_path', type=str, default='data/processed/augmented/chute_detection.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', type=bool, default=False, help='Include heatmaps in the training data')
    parser.add_argument('--inc_edges', type=bool, default=True, help='Include edges in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_wandb', action='store_true', help='Do not use Weights and Biases for logging')
    parser.add_argument('--model', type=str, default='vit', help='Model to use for training', choices=['cnn', 'convnextv2', 'vit', 'eva02'])
    parser.add_argument('--image_size', type=tuple, default=(1370, 204), help='Image size for the model (only for CNN)')
    parser.add_argument('--num_classes_straw', type=int, default=11, help='Number of classes for the straw classifier (11 = 10%, 21 = 5%)')
    parser.add_argument('--cont', action='store_true', help='Set model to predict a continuous value instead of a class (only for CNN model currently)')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--data_subsample', type=float, default=1.0, help='Amount of the data to subsample for training (1.0 = 100%, 0.5 = 50%)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    
    if torch.cuda.is_available(): print('Using GPU')
    
    match args.model:
        case 'cnn':
            image_size = args.image_size
        case 'convnextv2':
            image_size = (224, 224)
        case 'vit':
            image_size = (384, 384)
        case 'eva02':
            image_size = (448, 448)
    
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    train_set = dl.Chute(data_path=args.data_path, data_type='train', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
                         random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
                         num_classes_straw=args.num_classes_straw, continuous=args.cont, subsample=args.data_subsample)
    # test_set = dl.Chute(data_path=args.data_path, data_type='val', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges,
    #                     random_state=args.seed, force_update_statistics=False, data_purpose='straw', image_size=image_size, 
    #                     num_classes_straw=args.num_classes_straw, continuous=args.cont)
    
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accuracies = []
    fold_val_accuracies = []
    best_accuracies = []
    OVERALL_BEST_ACCURACY = -1.0 if not args.cont else torch.inf
    
    # WANDB
    if not args.no_wandb:
        initialize_wandb(args)
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
        print(f'Training fold {fold+1}/{args.folds}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_idx))
        test_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_idx))
        
        input_channels = 3
        if args.inc_heatmap: input_channels += 3
        if args.inc_edges: input_channels += 1
        
        # TRY: Using only the edge image as input
        # input_channels = 1
        
        
        if args.cont:
            args.num_classes_straw = 1
        
        match args.model:
            case 'cnn':
                model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=args.num_classes_straw)
            case 'convnextv2':
                model = timm.create_model('convnextv2_atto', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=True)
            case 'vit':
                model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=True)
            case 'eva02':
                model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=True)
        
        if args.cont and args.model != 'cnn':
            feature_shape = model.forward_features(torch.randn(1, input_channels, image_size[0], image_size[1])).shape
            feature_size = feature_shape[1] * feature_shape[2]
            feature_regressor = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1)
        else:
            feature_regressor = None
        
        train_loss, val_loss, train_accuracy, val_accuracy, best_accuracy = train_model(args, model, train_loader, test_loader, feature_regressor)

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_accuracies.append(train_accuracy)
        fold_val_accuracies.append(val_accuracy)
        best_accuracies.append(best_accuracy)
        
        # if args.cont:
        #     if np.mean(val_loss) < OVERALL_BEST_ACCURACY:
        #         OVERALL_BEST_ACCURACY = np.mean(val_loss)
        # else:
        #     if best_accuracy > OVERALL_BEST_ACCURACY:
        #         OVERALL_BEST_ACCURACY = best_accuracy
    
    mean_best_val_loss = np.mean(fold_val_losses)
    std_best_val_loss = np.std(fold_val_losses)
    
    if not args.cont:
        mean_best_val_accuracy = np.mean(best_accuracies)
        std_best_val_accuracy = np.std(best_accuracies)
        print(f'Mean Best Validation Accuracy: {mean_best_val_accuracy:.2f}% +/- {std_best_val_accuracy:.2f}%')
    
    print(f'Mean Best Validation Loss: {mean_best_val_loss:.4f} +/- {std_best_val_loss:.4f}')
    
    if not args.no_wandb:
        wandb.log({'mean_best_val_loss': mean_best_val_loss, 'std_best_val_loss': std_best_val_loss,
                   'mean_best_val_accuracy': mean_best_val_accuracy, 'std_best_val_accuracy': std_best_val_accuracy})
    
    model_folder = f'{args.model}_classifier/' if not args.cont else f'{args.model}_regressor/'
    for i in range(args.folds):
        x = np.arange(1, args.epochs+1, step=1)
        plt.plot(x, fold_train_losses[i], c='b', linestyle='--', alpha=0.2)
        plt.plot(x, fold_val_losses[i], c='r', alpha=0.2)
        
    plt.plot(x, np.mean(fold_train_losses, axis=0), c='b', label='Train Loss', linestyle='--')
    plt.plot(x, np.mean(fold_val_losses, axis=0), c='r', label='Val Loss')
    plt.ylim(0, min(7, np.max(fold_val_losses), np.max(fold_val_losses))+0.2)
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
        
        plt.plot(x, np.mean(fold_train_accuracies, axis=0), c='b', label='Train Loss', linestyle='--')
        plt.plot(x, np.mean(fold_val_accuracies, axis=0), c='r', label='Val Loss')
        plt.ylim(0, 1)
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