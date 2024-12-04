import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import timeit
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import linregress
import os

import data.dataloader as dl
import strawml.models.straw_classifier.cnn_classifier as cnn
import strawml.models.straw_classifier.feature_model as feature_model


def predict_model(args, model: torch.nn.Module, dataloader: DataLoader, feature_regressor: torch.nn.Module = None) -> tuple:
    """Predict the output of a model on a dataset.
    """
    
    if torch.cuda.is_available():
        print("Using GPU")
        model = model.cuda()
        if feature_regressor is not None:
            feature_regressor = feature_regressor.cuda()
    
    if args.cont:
        loss_fn = torch.nn.functional.mse_loss
    else:
        loss_fn = torch.nn.functional.cross_entropy
    
    model.eval()
    with torch.no_grad():
        data_iterator = tqdm(dataloader, desc="Predicting", unit="batch", position=0, leave=False)
        data_iterator.set_description(f"Predicting on {dataloader.dataset.data_type} data")
        
        accuracies = np.array([])
        losses = np.array([])
        outputs = np.array([])
        fullnesses = np.array([])
        sensor_data = np.array([])
        
        if dataloader.dataset.sensor:
            for (frame_data, target, sensor_fullness) in data_iterator:
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
                outputs = np.append(outputs, output.cpu().numpy())
                fullnesses = np.append(fullnesses, fullness.cpu().numpy())
                sensor_data = np.append(sensor_data, sensor_fullness.cpu().numpy())
            
            return outputs, fullnesses, accuracies, losses, sensor_data
        else:
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
                outputs = np.append(outputs, output.cpu().numpy())
                fullnesses = np.append(fullnesses, fullness.cpu().numpy())
    
            return outputs, fullnesses, accuracies, losses


def plot_cont_predictions(outputs, fullnesses, sensor_data=None):
    """Plot the continuous predictions.
    """
    
    outputs = np.array(outputs)
    fullnesses = np.array(fullnesses)
    if sensor_data is not None:
        sensor_data = np.array(sensor_data)
    
    # MSE = np.square(np.subtract(outputs, fullnesses)).mean()
    # MAE = np.abs(np.subtract(outputs, fullnesses)).mean()
    # RMSE = np.sqrt(MSE)
    # PRMSE = np.sqrt(np.mean(np.sum(np.square(np.subtract(outputs, fullnesses)))) / np.sum(np.square(fullnesses)))
    
    # Calculate accuracy of the model
    # Any prediction within x% of the true value is considered correct
    acceptable = 0.1
    accuracy = np.sum(np.abs(outputs - fullnesses) < acceptable) / len(outputs)
    sensor_accuracy = np.sum(np.abs(sensor_data - fullnesses) < acceptable) / len(sensor_data)
    
    model_name = f'{args.model}_reg' if args.cont else f'{args.model}_cls'
    
    plt.figure(figsize=(10, 5))
    plt.plot(outputs, label='Predicted', color='indianred', linestyle='--')
    if sensor_data is not None:
        plt.plot(sensor_data, label='Sensor', color='darkgreen', linestyle='-.')
    plt.plot(fullnesses, label='True', color='royalblue', alpha=0.75)
    
    # Plot acceptable range
    plt.fill_between(np.arange(len(outputs)), fullnesses-acceptable, fullnesses+acceptable, color='gray', alpha=0.5, label=f'threshold={acceptable*100:.0f}%')
    
    plt.legend()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    plt.xlabel('Frame')
    plt.ylabel('Fullness')
    # plt.title(f'{model_name} Predicted vs True Fullness, MSE: {MSE:.3f}, MAE: {MAE:.3f}, RMSE: {RMSE:.3f}, PRMSE: {PRMSE:.3f}, accuracy: {accuracy*100:.1f}%')
    if sensor_data is not None:
        plt.title(f'{model_name} Predicted vs True Fullness, model accuracy: {accuracy*100:.1f}%, sensor accuracy: {sensor_accuracy*100:.1f}%')
    else:
        plt.title(f'{model_name} Predicted vs True Fullness, model accuracy: {accuracy*100:.1f}%')
    plt.tight_layout()
    os.makedirs(f'reports/figures/model_results/{args.model}_regressor/', exist_ok=True)
    plt.savefig(f'reports/figures/model_results/{args.model}_regressor/cont_predictions.png', dpi=300)
    plt.show()
    
    
    # Plot regression fit
    plt.figure(figsize=(10, 5))
    # adding the regression line to the scatter plot
    slope, intercept, r_value, p_value, std_err = linregress(fullnesses, outputs)
    plt.plot(fullnesses, slope*fullnesses + intercept, color='royalblue', label=f'Regression Fit ' + r"($R^2 = $" + f"{r_value**2:.2f})", zorder=2)
    # Plot targets and predictions
    plt.scatter(fullnesses, outputs, color='indianred', alpha=.75, label='Predicted vs True', zorder=0)
    # Plot 1:1 line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, alpha=.75, zorder=1)
    plt.xlabel('True Fullness')
    plt.ylabel('Predicted Fullness')
    plt.title(f'{model_name} Predicted vs True Fullness')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/model_results/{args.model}_regressor/cont_predictions_scatter.png', dpi=300)
    plt.show()


def plot_roc(outputs, fullness):
    """Plots the ROC curve for the model.
    """
    
    fpr, tpr, thresholds = roc_curve(fullness, outputs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('reports/figures/straw_analysis/model_results/roc.png', dpi=300)
    plt.show()
    
    

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
    plt.title(f'Confusion Matrix, accuracy: {accuracy*100:.1f}%')
    # Set xticks and yticks with labels
    plt.xticks(ticks=np.arange(num_classes), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(num_classes), labels=labels)
    
    # Plot numbers on each square
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black' if confusion_matrix[i, j] > np.max(confusion_matrix)/2 else 'white')
    plt.tight_layout()
    plt.savefig('reports/figures/straw_analysis/model_results/confusion_matrix.png', dpi=300)
    plt.show()
    


def convert_class_to_fullness(class_num):
    """Convert the class number to the fullness value.
    """
    return (class_num + 1) * 5


def get_args() -> argparse.Namespace:
    """Get the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description='Train the CNN classifier model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default='models/', help='Path to load the model from')
    parser.add_argument('--data_path', type=str, default='data/processed/sensors.hdf5', help='Path to the training data')
    parser.add_argument('--inc_heatmap', type=bool, default=False, help='Include heatmaps in the training data')
    parser.add_argument('--inc_edges', type=bool, default=False, help='Include edges in the training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='convnextv2', help='Model to use for predicting', choices=['cnn', 'convnextv2', 'vit', 'eva02', 'caformer'])
    parser.add_argument('--image_size', type=tuple, default=(224, 224), help='Image size for the model (only for CNN)')
    parser.add_argument('--num_classes_straw', type=int, default=11, help='Number of classes for the straw classifier (11 = 10%, 21 = 5%)')
    parser.add_argument('--cont', action='store_true', help='Set model to predict a continuous value instead of a class (only for CNN model currently)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
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
    
    temp_set = dl.Chute(data_path='data/processed/train.hdf5', data_type='train', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges, image_size=image_size,
                        random_state=args.seed, force_update_statistics=False, data_purpose='straw',
                        num_classes_straw=args.num_classes_straw, continuous=args.cont)
    mean, std = temp_set.train_mean, temp_set.train_std
    if args.inc_heatmap:
        hm_mean, hm_std = temp_set.train_hm_mean, temp_set.train_hm_std
    
    statistics = (mean, std) if not args.inc_heatmap else (mean, std, hm_mean, hm_std)
    
    test_set = dl.Chute(data_path=args.data_path, data_type='test', inc_heatmap=args.inc_heatmap, inc_edges=args.inc_edges, image_size=image_size,
                        random_state=args.seed, force_update_statistics=False, data_purpose='straw',
                        num_classes_straw=args.num_classes_straw, continuous=args.cont, override_statistics=statistics, sensor=True)
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    input_channels = 3
    if args.inc_heatmap:
        input_channels += 3
    if args.inc_edges:
        input_channels += 1
    
    if args.cont:
        args.num_classes_straw = 1
    
    match args.model:
        case 'cnn':
            model = cnn.CNNClassifier(image_size=image_size, input_channels=input_channels, output_size=args.num_classes_straw)
        case 'convnextv2':
            model = timm.create_model('convnext_small.in12k_ft_in1k_384', pretrained=False, in_chans=input_channels, num_classes=args.num_classes_straw)
        case 'vit':
            model = timm.create_model('vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', pretrained=False, in_chans=input_channels, num_classes=args.num_classes_straw)
        case 'eva02':
            model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=False, in_chans=input_channels, num_classes=args.num_classes_straw)
        case 'caformer':
            model = timm.create_model('caformer_m36.sail_in22k_ft_in1k_384', in_chans=input_channels, num_classes=args.num_classes_straw, pretrained=False)
    

    model_type = 'regressor' if args.cont else 'classifier'
    model_path = f'{args.model_path}{args.model}_{model_type}/'
    if args.cont and args.model != 'cnn':
        features = model.forward_features(torch.randn(1, input_channels, image_size[0], image_size[1]))
        feature_size = torch.flatten(features, 1).shape[1]
        feature_regressor = feature_model.FeatureRegressor(image_size=image_size, input_size=feature_size, output_size=1)
        
        model.load_state_dict(torch.load(f'{model_path}/{args.model}_feature_extractor_best.pth', weights_only=True))
        feature_regressor.load_state_dict(torch.load(f'{model_path}/{args.model}_regressor_best.pth', weights_only=True))
        
    else:
        feature_regressor = None
        model.load_state_dict(torch.load(f'{model_path}/{args.model}_classifier_best', weights_only=True))
    
    outputs, fullnesses, accuracies, losses, sensor_data = predict_model(args, model, test_loader, feature_regressor=feature_regressor)
    
    print(f"Mean loss: {sum(losses) / len(losses)}")
    if not args.cont:
        print(f"Mean accuracy: {sum(accuracies) / len(accuracies)}")
    else:
        print(f"Mean output: {sum(outputs) / len(outputs)}")
        print(f"Mean fullness: {sum(fullnesses) / len(fullnesses)}")
    
    print("Done predicting.")
    
    if args.cont:
        plot_cont_predictions(outputs, fullnesses, sensor_data)
    else:
        plot_confusion_matrix(outputs, fullnesses, args.num_classes_straw)