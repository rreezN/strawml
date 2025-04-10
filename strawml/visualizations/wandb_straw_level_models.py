import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyse_model_selection():
    list_of_csvs = os.listdir("reports/")
    list_of_csvs = [x for x in list_of_csvs if x.endswith(".csv")]
    
    
    print("############### MODEL SELECTION ###############")
    for csv in list_of_csvs:
        if "hyperparameter" in csv:
            continue
        current_model_name = csv.split(" ")[0]
        print(f' ----------------- {current_model_name} ----------------- ')
        df = pd.read_csv("reports/" + csv, sep=',', index_col='Name')
        df = df.drop(columns=["_wandb"])
        
        # Get the means of each row
        df["mean"] = df.mean(numeric_only=True, axis=1)
        df = df.sort_values(by="mean", ascending=False)
        
        # Get the mins of each row
        df["min"] = df.min(numeric_only=True, axis=1)
        
        # Get the maxs of each row
        df["max"] = df.max(numeric_only=True, axis=1)
        
        # Get the stds of each row
        df["std"] = df.std(numeric_only=True, axis=1)
        
        # Print the top 5 models and their mean, min, max, and std
        print("Top 5 models: ")
        for i in range(5):
            print(f'{df.index[i]}: {df["mean"].iloc[i]*100:.2f}% \t\t\t (min: {df["min"].iloc[i]*100:.2f}%, max: {df["max"].iloc[i]*100:.2f}%, std: {df["std"].iloc[i]*100:.2f}%)')
        
        # Compare large image sizes with small image sizes
        # print("\nComparing large image sizes with small image sizes: ")
        # for model in df.index:
        #     if "large" or "L" or "224" or "384" in model:
        #         print(f'{model}: {df["mean"].loc[model]*100:.2f}% \t\t\t (min: {df["min"].loc[model]*100:.2f}%, max: {df["max"].loc[model]*100:.2f}%, std: {df["std"].loc[model]*100:.2f}%)')

        # Compare large images with layers
        # print("\nComparing large images with layers: ")
        # for model in df.index:
        #     if 'large' or 'layers' in model:
        #         print(f'{model}: {df["mean"].loc[model]*100:.2f}% \t\t\t (min: {df["min"].loc[model]*100:.2f}%, max: {df["max"].loc[model]*100:.2f}%, std: {df["std"].loc[model]*100:.2f}%)')
        
        
    print("\n\n############### HYPERPARAMETERS ###############")
    for csv in list_of_csvs:
        if not "hyperparameter" in csv:
            continue
        current_model_name = csv.split(" ")[0]
        print(f' ----------------- {current_model_name} ----------------- ')
        df = pd.read_csv("reports/" + csv, sep=',', index_col='Name')
        df = df.drop(columns=["_wandb"])
        
        # Get the means of each row
        df["mean"] = df.mean(numeric_only=True, axis=1)
        df = df.sort_values(by="mean", ascending=False)
        
        # Get the mins of each row
        df["min"] = df.min(numeric_only=True, axis=1)
        
        # Get the maxs of each row
        df["max"] = df.max(numeric_only=True, axis=1)
        
        # Get the stds of each row
        df["std"] = df.std(numeric_only=True, axis=1)
        
        # Print the top 5 models and their mean, min, max, and std
        for i in range(5):
            print(f'{df.index[i]}: {df["mean"].iloc[i]*100:.2f}% \t\t\t (min: {df["min"].iloc[i]*100:.2f}%, max: {df["max"].iloc[i]*100:.2f}%, std: {df["std"].iloc[i]*100:.2f}%)')
        
def plot_training_curves():
    list_of_csvs = os.listdir("reports/training curves")
    list_of_csvs = [x for x in list_of_csvs if x.endswith(".csv")]

    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax = ax.ravel()
    
    for j, csv in enumerate(list_of_csvs):
        df = pd.read_csv("reports/training curves/" + csv, sep=',')
        
        model_names = []
        for column in df.columns[1:]:
            column_model_name = column.split(" ")[0]
            if column_model_name not in model_names:
                model_names.append(column_model_name)
        
        training_losses = {}
        validation_losses = {}
        for model_name in model_names:
            training_losses[model_name] = {}
            validation_losses[model_name] = {}
            for i in range(4):
                training_losses[model_name].update({f'f{i+1}': df[f'{model_name} - f{i+1}_train_loss'].tolist()})
                validation_losses[model_name].update({f'f{i+1}': df[f'{model_name} - f{i+1}_val_loss'].tolist()})
                
        # Calculate the mean of the training and validation losses
        for model_name in model_names:
            training_losses[model_name]['mean'] = np.mean([training_losses[model_name][f'f{i+1}'] for i in range(4)], axis=0)
            validation_losses[model_name]['mean'] = np.mean([validation_losses[model_name][f'f{i+1}'] for i in range(4)], axis=0)
            
            training_losses[model_name]['min'] = np.min([training_losses[model_name][f'f{i+1}'] for i in range(4)], axis=0)
            validation_losses[model_name]['min'] = np.min([validation_losses[model_name][f'f{i+1}'] for i in range(4)], axis=0)
            
            training_losses[model_name]['max'] = np.max([training_losses[model_name][f'f{i+1}'] for i in range(4)], axis=0)
            validation_losses[model_name]['max'] = np.max([validation_losses[model_name][f'f{i+1}'] for i in range(4)], axis=0)
        
        # Plot the training and validation losses
        colors = ['goldenrod', 'mediumseagreen', 'darkorchid', 'lightsalmon', 'rebeccapurple']
        x = df['custom_step']
        for i, model_name in enumerate(model_names):
            short_name = model_name.split("-")[1:]
            short_name = "-".join(short_name)
            
            # set alpha of the lines
            if short_name == 'sweep-1' or short_name == 'sweep-10':
                alpha = 0.75
                if short_name == 'sweep-1': color = 'indianred'
                elif short_name == 'sweep-10': color = 'royalblue'
                short_name += ' (best)'
            else:
                alpha = 0.25
                color = colors[i]
                
            ax[j].plot(x, training_losses[model_name]['mean'], label=f'{short_name} - train', color=color, alpha=alpha)
            ax[j].plot(x, validation_losses[model_name]['mean'], label=f'{short_name} - val', color=color, linestyle='dashed', alpha=alpha)
        
        
        font_size = 20
        ax[j].set_title(f'{csv.split(" ")[0]} - Training and Validation Losses', fontsize=font_size)
        ax[j].set_xlabel('Epoch', fontsize=font_size)
        ax[j].set_xticks = np.array(x.tolist(), dtype=int)
        ax[j].tick_params(axis='both', labelsize=font_size-6)
        ax[j].set_ylabel('Loss (MSE)', fontsize=font_size)
        ax[j].set_ylim(0, 0.0035)
        ax[j].grid()
        ax[j].legend(fontsize=font_size-8, ncol=2)#len(model_names))
    
    plt.tight_layout()
    plt.savefig("reports/training curves/straw_model_training_curves.pdf", dpi=300, bbox_inches='tight')

def plot_hyper_parameters_vs_accuracy():
    # df = pd.DataFrame(columns=['model', 'sweep', 'accuracy', 'learning_rate', 'batch_size'])
    # df['model'] = ['ConvNeXtV1'] * 5 + ['ConvNeXtV2'] * 5
    # df['sweep'] = ['sweep-1', 'sweep-15', 'sweep-26', 'sweep-27', 'sweep-28', 'sweep-9', 'sweep-10', 'sweep-21', 'sweep-30', 'sweep-31']
    # df['accuracy'] = [93.37, 92.53, 92.83, 93.33, 93.20, 94.43, 94.98, 94.30, 94.34, 94.84]
    # df['learning_rate'] = [0.00001120659857537586,
    #                        0.00006481605040715890,
    #                        0.00001655146758340562,
    #                        0.00008164009822514294,
    #                        0.00001953371338081623,
    #                        0.00004468313613877140,
    #                        0.00003012743634235883,
    #                        0.00004732196011865191,
    #                        0.00001167650323466638,
    #                        0.00001835140223882227]
    # df['batch_size'] = [4, 8, 4, 4, 4, 12, 12, 8, 4, 4]
    
    # # Plot the learning rate vs accuracy
    # plt.figure(figsize=(10, 5))
    
    # v1_col = 'indianred'
    # v2_col = 'royalblue'
    
    # fontsize = 20
    # for i in range(len(df)):
    #     match df['batch_size'].iloc[i]:
    #         case 4:
    #             marker = 'x'
    #             markersize = 200
    #         case 8:
    #             marker = 'o'
    #             markersize = 400
    #         case 12:
    #             marker = 's'
    #             markersize = 600
            
    #     if df['model'].iloc[i] == 'ConvNeXtV1':
    #         plt.scatter(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], color=v1_col, label='ConvNeXtV1' if i == 0 else None, s=markersize, alpha=0.5)
    #     else:
    #         plt.scatter(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], color=v2_col, label='ConvNeXtV2' if i == 5 else None, s=markersize, alpha=0.5)

    #     # plot batch size as text
    #     plt.text(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], f'{df["batch_size"].iloc[i]}', fontsize=fontsize-4, ha='center', va='center', color='black')
        
    
    # plt.xlabel('Learning Rate', fontsize=fontsize)
    # plt.ylabel('Accuracy (%)', fontsize=fontsize)
    # plt.title('Learning Rate vs Accuracy', fontsize=fontsize)
    # plt.grid()
    # plt.legend(ncol=2, fontsize=fontsize-2)
    # plt.tick_params(axis='both', labelsize=fontsize-4)
    # plt.tight_layout()
    # plt.savefig('reports/figures/learning_rate_vs_accuracy.pdf', dpi=300, bbox_inches='tight')
    
    list_of_csvs = os.listdir("reports/straw sweep all")
    list_of_csvs = [x for x in list_of_csvs if x.endswith(".csv")]
    
    plt.figure(figsize=(15, 5))
    fontsize = 20
    for j, csv in enumerate(list_of_csvs):
        df = pd.read_csv("reports/straw sweep all/" + csv, sep=',')
        
        accuracies = []
        for sweep in df['Name']:
            accuracies = np.zeros((4, len(df)))
            for i in range(1, 5):
                accuracies[i-1] = np.array(df[f'f{i} mean sensor prediction accuracies'].tolist())
            mean_accuracies = np.mean(accuracies, axis=0)
        
        df['accuracy'] = mean_accuracies*100
       
        # Plot the learning rate vs accuracy
        col = 'indianred' if j == 0 else 'royalblue'
        for i in range(len(df)):
            match df['batch_size'].iloc[i]:
                case 4:
                    marker = 'x'
                    markersize = 200
                case 8:
                    marker = 'o'
                    markersize = 400
                case 12:
                    marker = 's'
                    markersize = 600
            
            if 'toasty-sweep-10' in df['Name'].iloc[i] or 'summer-sweep-1' in df['Name'].iloc[i]:
                edgecolor = 'black'
                linewidth = 2
                alpha = 1.0
                zorder = 10
            else:
                edgecolor = None
                linewidth = 1
                alpha = 0.5
                zorder = 1
            
            if i == 0:
                plt.scatter(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], color=col, label='ConvNeXtV1' if j == 0 else 'ConvNeXtV2', s=200, alpha=alpha, linewidth=linewidth, edgecolor=edgecolor, zorder=zorder)
            else:
                plt.scatter(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], color=col, s=markersize, alpha=alpha, linewidth=linewidth, edgecolor=edgecolor, zorder=zorder)
            
            # plot batch size as text
            # plt.text(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], f'{df["batch_size"].iloc[i]}', fontsize=fontsize-4, ha='center', va='center', color='black')
            
    plt.xlabel('Learning Rate', fontsize=fontsize)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Accuracy (%)', fontsize=fontsize)
    plt.title('Learning Rate vs Accuracy', fontsize=fontsize)
    plt.grid()
    plt.legend(ncol=2, fontsize=fontsize-2)
    plt.tick_params(axis='both', labelsize=fontsize-4)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_rate_vs_accuracy.pdf', dpi=300, bbox_inches='tight')
            

def plot_noisy_data_curves():
    yolo_frame = [100, 99.9, 98.9, 98.1, 95.5, 93.1, 87]
    yolo_acc = [94.52, 89.99, 83.14, 79.699, 76.23, 69.748, 63.1]
    yolo_mae = [2.92, 4.378, 6.034, 8.907, 10.386, 13.793, 18.853]
    yolo_text_offset = [5, 5, 5, 5, 5, 5, 5]
    
    convnextv1_frame = [100, 100, 100, 99.7, 99.6, 99.5, 99.4]
    convnextv1_acc = [92.78, 85.565, 79.478, 74.171, 66.987, 64.248, 59.93]
    convnextv1_mae = [4.64, 6.191, 8.193, 9.831, 11.459, 12.185, 13.151]
    convnextv1_text_offset = [-7, -6, -6, -6, -6, -6, -6]
    
    convapril_frame = [100, 100, 91.4, 81.6, 79.7, 77.2, 71.8]
    convapril_acc = [75.391, 73.217, 61.464, 52.345, 48.201, 39.077, 19.734]
    convapril_mae = [6.442, 7.413, 10.86, 16.931, 18.341, 27.104, 36.803]
    convapril_text_offset = [-7, -10, -7, -7, -6, 7, 7]
    
    model_names = ['YOLO-S', 'ConvNeXtV1', 'ConvNeXtA']
    model_colors = ['royalblue', 'indianred', 'seagreen']
    
    all_acc = [yolo_acc, convnextv1_acc, convapril_acc]
    all_mae = [yolo_mae, convnextv1_mae, convapril_mae]
    all_frame = [yolo_frame, convnextv1_frame, convapril_frame]
    all_text_offset = [yolo_text_offset, convnextv1_text_offset, convapril_text_offset]
    
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    axes = ax.ravel()
    fontsize = 18
    
    x = [0, 1, 2, 3, 4, 5, 6]
    
    for i in range(len(model_names)):
        # Plot the accuracy
        axes[0].plot(x, all_acc[i], label=f'{model_names[i]} Acc', color=model_colors[i], marker='o')
        for j in range(len(all_acc[i])):
            axes[0].text(j, all_acc[i][j] +all_text_offset[i][j], f'{all_acc[i][j]:.2f}%', fontsize=fontsize-6, ha='center', va='center', color='black')
    
        # Plot the mae
        axes[0].plot(x, all_mae[i], label=f'{model_names[i]} MAE', color=model_colors[i], marker='o', linestyle='dashed')
        
        # Plot the frame accuracy
        axes[1].plot(x, all_frame[i], label=f'{model_names[i]} Frames', color=model_colors[i], marker='o', linestyle='dotted')

    # Set the labels and titles
    axes[0].set_ylim(-10, 110)
    axes[1].set_ylim(65, 105)
    axes[1].set_xlabel('# Noisy Transformations', fontsize=fontsize)
    axes[0].set_ylabel('MAE | Accuracy', fontsize=fontsize)
    axes[1].set_ylabel('% Frames', fontsize=fontsize)
    axes[0].set_title('Performance vs Noisy Transformations', fontsize=fontsize)
    axes[0].legend(fontsize=fontsize-4, loc='lower center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=3)
    axes[1].legend(fontsize=fontsize-4, loc='lower center', bbox_to_anchor=(0.5, -1), fancybox=True, shadow=True, ncol=3)
    axes[0].tick_params(axis='both', labelsize=fontsize-6)
    axes[1].tick_params(axis='both', labelsize=fontsize-6)
    axes[0].grid()
    axes[1].grid()
    plt.tight_layout()
    
    # fig.tick_params(axis='both', labelsize=fontsize-6)
    # plt.xlabel('# Noisy Transformations', fontsize=fontsize)
    # plt.ylabel('MAE | Accuracy | Frame Accuracy', fontsize=fontsize)
    # plt.title('Performance vs Noisy Transformations', fontsize=fontsize)
    # plt.legend(fontsize=fontsize-4, ncols=len(model_names), loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=True, shadow=True)
    # plt.ylim(-5, 105)
    # plt.grid()
    plt.savefig('reports/figures/noisy_transformations_results.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    
        

def plot_final_run_curves():
    list_of_csvs = os.listdir("reports/training curves/final run")
    list_of_csvs = [x for x in list_of_csvs if x.endswith(".csv")]

    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax = ax.ravel()
    
    for j, csv in enumerate(list_of_csvs):
        df = pd.read_csv("reports/training curves/final run/" + csv, sep=',')
        
        model_names = []
        for column in df.columns[1:]:
            column_model_name = column.split(" ")[0]
            if column_model_name not in model_names:
                model_names.append(column_model_name)
        
        training_losses = {}
        validation_losses = {}
        for model_name in model_names:
            training_losses[model_name] = df[f'{model_name} - train_loss'].tolist()
            validation_losses[model_name] = df[f'{model_name} - val_loss'].tolist()
        
        v1_apriltag_best = 77
        v1_best = 68
        v2_best = 15
        
        best = [v1_apriltag_best, v1_best, v2_best]
        best_y_train = [training_losses['ConvNeXtV1-AprilTag'][v1_apriltag_best-1], training_losses['ConvNeXtV1'][v1_best-1], training_losses['ConvNeXtV2'][v2_best-1]]
        best_y_val = [validation_losses['ConvNeXtV1-AprilTag'][v1_apriltag_best-1], validation_losses['ConvNeXtV1'][v1_best-1], validation_losses['ConvNeXtV2'][v2_best-1]]
        
        
        # if best_y_train[0] < best_y_train[1]:
        #     text_offset_train = [0.000075, -0.000075]
        # else:
        #     text_offset_train = [-0.000075, 0.000075]
        
        # if best_y_val[0] < best_y_val[1]:
        #     text_offset_val = [0.003, -0.003]
        # else:
        #     text_offset_val = [-0.003, 0.003]
        
        # Get min and max values for text offset
        text_offset_train = [0.000075, -0.000075, 0.000075]
        text_offset_val = [0.003, -0.003, -0.003]
        
        
        # Plot the training and validation losses
        colors = ['seagreen', 'indianred', 'royalblue',]
        x = df['Step']
        font_size = 20
        for i, model_name in enumerate(model_names):
            # long_model_name = 'ConvNeXtV1' if 'v1' in model_name else 'ConvNeXtV2'
            # plot training loss
            ax[0].plot(x, training_losses[model_name], label=f'{model_name} - train', color=colors[i])
            # plot validation loss
            ax[1].plot(x, validation_losses[model_name], label=f'{model_name} - val', color=colors[i], linestyle='dashed')
            
            # Plot point where best model was selected
            ax[0].scatter(best[i], best_y_train[i], color=colors[i], marker='.', s=150, zorder=10)
            ax[1].scatter(best[i], best_y_val[i], color=colors[i], marker='.', s=150, zorder=10)
            
            # plot the best model loss above the point
            ax[0].text(best[i], best_y_train[i]+text_offset_train[i], f'{best_y_train[i]:.5f}', fontsize=font_size-6, ha='center', va='center',
                       bbox=dict(facecolor='white', edgecolor=colors[i], alpha=0.5, boxstyle='round'))
            ax[1].text(best[i], best_y_val[i]+text_offset_val[i], f'{best_y_val[i]:.4f}', fontsize=font_size-6, ha='center', va='center',
                       bbox=dict(facecolor='white', edgecolor=colors[i], alpha=0.5, boxstyle='round'))
        
        
        ax[0].set_title(f'Training', fontsize=font_size)
        ax[1].set_title(f'Validation', fontsize=font_size)
        for axis in ax:
            axis.set_xlabel('Epoch', fontsize=font_size)
            axis.set_xticks = np.array(x.tolist(), dtype=int)
            axis.tick_params(axis='both', labelsize=font_size-6)
            axis.set_ylabel('Loss (MSE)', fontsize=font_size)
            # axis.set_ylim(0, 0.0035)
            axis.grid()
            axis.legend(fontsize=font_size-8, ncol=1)
    
    fig.suptitle('Straw Level Models Final Training and Validation Losses', fontsize=font_size)
    
    plt.tight_layout()
    plt.savefig("reports/training curves/final run/straw_model_final_run.pdf", dpi=300, bbox_inches='tight')

def create_table_of_model():
    df = pd.read_csv('reports/straw model selection all/straw_level_models_model_selection_all_data.csv', sep=',', index_col='Name')
    
    # df = df[df['State'] == 'finished']
    df = df[df['data_subsample'] == 1.0]
    # df = df.drop(columns=['State', 'User', 'Created'])
    df = df.drop_duplicates()
    
    df['augment_probability'] = df['augment_probability'].fillna(0.0)
    df['balanced_dataset'] = df['balanced_dataset'].fillna(False)
    df['inc_heatmap'] = df['inc_heatmap'].fillna(False)
    df['inc_edges'] = df['inc_edges'].fillna(False)
    df['only_head'] = df['only_head'].fillna(False)
    df['use_sigmoid'] = df['use_sigmoid'].fillna(False)
    df['continuous'] = df['continuous'].fillna(False)
    df['num_hidden_layers'] = df['num_hidden_layers'].fillna(0)
    df['num_neurons'] = df['num_neurons'].fillna(512)
    
    df['augment_probability'] = df['augment_probability'] * 100
    
    for model in df.index:
        mean_accuracy = 0
        divisor = 0
        for i in range(4):
            mean_accuracy += df[f'f{i+1} mean sensor prediction accuracies'].loc[model] if not np.isnan(df[f'f{i+1} mean sensor prediction accuracies'].loc[model]) else 0
            divisor += 1 if not np.isnan(df[f'f{i+1} mean sensor prediction accuracies'].loc[model]) else 0
        mean_accuracy /= divisor
        if mean_accuracy <= 1.0:
            mean_accuracy *= 100
        print(f'{model}: {mean_accuracy:.2f}%')
        df.loc[model, 'mean_accuracy'] = mean_accuracy
    
            
    df = df.sort_values(by='mean_accuracy', ascending=False)    

    # Print markdown table of the models
    print('----- ViT Model Selection -----')
    vit_df = df[df['model'] == 'vit']
    vit_df = vit_df.sort_values(by='mean_accuracy', ascending=False)
    vit_df['continuous'] = vit_df['continuous'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['balanced_dataset'] = vit_df['balanced_dataset'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['inc_heatmap'] = vit_df['inc_heatmap'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['inc_edges'] = vit_df['inc_edges'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['only_head'] = vit_df['only_head'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['use_sigmoid'] = vit_df['use_sigmoid'].map({True: '$\checkmark$', False: '$\\times$'})
    print(f'| Model | Accuracy | Image Size | Regression | Heatmap | Sigmoid | Head Hidden Layers |')
    print(f'| --- | --- | --- | --- | --- | --- | --- |')
    print(f'|  | Optimiser | Augment Probability | Balanced Dataset | Edges | Only Head | Head Layer Size |')
    for model in vit_df.index:
        print(f'| {model} | {vit_df["mean_accuracy"].loc[model]:.2f}\% | {vit_df["image_size"].loc[model]} | {vit_df["continuous"].loc[model]} | {vit_df["inc_heatmap"].loc[model]} | {vit_df["use_sigmoid"].loc[model]} | {vit_df["num_hidden_layers"].loc[model]:.0f} |')
        print(f'| | {vit_df["optim"].loc[model]} | {vit_df["augment_probability"].loc[model]:.0f}\% | {vit_df["balanced_dataset"].loc[model]} | {vit_df["inc_edges"].loc[model]} | {vit_df["only_head"].loc[model]} | {vit_df["num_neurons"].loc[model]:.0f} |')

    print()
    print('----- ConvNeXtV1 Model Selection -----')
    vit_df = df[df['model'].isin(['convnext', 'convnextv2'])]
    vit_df = vit_df[vit_df['Tags'] == 'v1']
    vit_df = vit_df.sort_values(by='mean_accuracy', ascending=False)
    vit_df['continuous'] = vit_df['continuous'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['balanced_dataset'] = vit_df['balanced_dataset'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['inc_heatmap'] = vit_df['inc_heatmap'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['inc_edges'] = vit_df['inc_edges'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['only_head'] = vit_df['only_head'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['use_sigmoid'] = vit_df['use_sigmoid'].map({True: '$\checkmark$', False: '$\\times$'})
    print(f'| Model | Accuracy | Image Size | Regression | Heatmap | Sigmoid | Head Hidden Layers |')
    print(f'| --- | --- | --- | --- | --- | --- | --- |')
    print(f'|  | Optimiser | Augment Probability | Balanced Dataset | Edges | Only Head | Head Layer Size |')
    for model in vit_df.index:
        print(f'| {model} | {vit_df["mean_accuracy"].loc[model]:.2f}\% | {vit_df["image_size"].loc[model]} | {vit_df["continuous"].loc[model]} | {vit_df["inc_heatmap"].loc[model]} | {vit_df["use_sigmoid"].loc[model]} | {vit_df["num_hidden_layers"].loc[model]:.0f} |')
        print(f'| | {vit_df["optim"].loc[model]} | {vit_df["augment_probability"].loc[model]:.0f}\% | {vit_df["balanced_dataset"].loc[model]} | {vit_df["inc_edges"].loc[model]} | {vit_df["only_head"].loc[model]} | {vit_df["num_neurons"].loc[model]:.0f} |')
    
    print()
    print('----- ConvNeXtV2 Model Selection -----')
    vit_df = df[df['model'].isin(['convnextv2'])]
    vit_df = vit_df[vit_df['Tags'] != 'v1']
    vit_df = vit_df.sort_values(by='mean_accuracy', ascending=False)
    vit_df['continuous'] = vit_df['continuous'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['balanced_dataset'] = vit_df['balanced_dataset'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['inc_heatmap'] = vit_df['inc_heatmap'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['inc_edges'] = vit_df['inc_edges'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['only_head'] = vit_df['only_head'].map({True: '$\checkmark$', False: '$\\times$'})
    vit_df['use_sigmoid'] = vit_df['use_sigmoid'].map({True: '$\checkmark$', False: '$\\times$'})
    print(f'| Model | Accuracy | Image Size | Regression | Heatmap | Sigmoid | Head Hidden Layers |')
    print(f'| --- | --- | --- | --- | --- | --- | --- |')
    print(f'|  | Optimiser | Augment Probability | Balanced Dataset | Edges | Only Head | Head Layer Size |')
    for model in vit_df.index:
        print(f'| {model} | {vit_df["mean_accuracy"].loc[model]:.2f}\% | {vit_df["image_size"].loc[model]} | {vit_df["continuous"].loc[model]} | {vit_df["inc_heatmap"].loc[model]} | {vit_df["use_sigmoid"].loc[model]} | {vit_df["num_hidden_layers"].loc[model]:.0f} |')
        print(f'| | {vit_df["optim"].loc[model]} | {vit_df["augment_probability"].loc[model]:.0f}\% | {vit_df["balanced_dataset"].loc[model]} | {vit_df["inc_edges"].loc[model]} | {vit_df["only_head"].loc[model]} | {vit_df["num_neurons"].loc[model]:.0f} |')
    
    
if __name__ == '__main__':
    # analyse_model_selection()
    # plot_training_curves()
    # plot_hyper_parameters_vs_accuracy()
    # plot_final_run_curves()
    # create_table_of_model()
    plot_noisy_data_curves()