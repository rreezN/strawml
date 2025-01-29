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

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
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
        ax[j].legend(fontsize=font_size-8)
    
    plt.tight_layout()
    plt.savefig("reports/training curves/straw_model_training_curves.pdf", dpi=300, bbox_inches='tight')

def plot_hyper_parameters_vs_accuracy():
    df = pd.DataFrame(columns=['model', 'sweep', 'accuracy', 'learning_rate', 'batch_size'])
    df['model'] = ['ConvNeXtV1'] * 5 + ['ConvNeXtV2'] * 5
    df['sweep'] = ['sweep-1', 'sweep-15', 'sweep-26', 'sweep-27', 'sweep-28', 'sweep-9', 'sweep-10', 'sweep-21', 'sweep-30', 'sweep-31']
    df['accuracy'] = [93.37, 92.53, 92.83, 93.33, 93.20, 94.43, 94.98, 94.30, 94.34, 94.84]
    df['learning_rate'] = [0.00001120659857537586,
                           0.00006481605040715890,
                           0.00001655146758340562,
                           0.00008164009822514294,
                           0.00001953371338081623,
                           0.00004468313613877140,
                           0.00003012743634235883,
                           0.00004732196011865191,
                           0.00001167650323466638,
                           0.00001835140223882227]
    df['batch_size'] = [4, 8, 4, 4, 4, 12, 12, 8, 4, 4]
    
    # Plot the learning rate vs accuracy
    plt.figure(figsize=(10, 5))
    
    v1_col = 'indianred'
    v2_col = 'royalblue'
    
    fontsize = 20
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
            
        if df['model'].iloc[i] == 'ConvNeXtV1':
            plt.scatter(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], color=v1_col, label='ConvNeXtV1' if i == 0 else None, s=markersize, alpha=0.5)
        else:
            plt.scatter(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], color=v2_col, label='ConvNeXtV2' if i == 5 else None, s=markersize, alpha=0.5)

        # plot batch size as text
        plt.text(df['learning_rate'].iloc[i], df['accuracy'].iloc[i], f'{df["batch_size"].iloc[i]}', fontsize=fontsize-4, ha='center', va='center', color='black')
        
    
    plt.xlabel('Learning Rate', fontsize=fontsize)
    plt.ylabel('Accuracy (%)', fontsize=fontsize)
    plt.title('Learning Rate vs Accuracy', fontsize=fontsize)
    plt.grid()
    plt.legend(ncol=2, fontsize=fontsize-2)
    plt.tick_params(axis='both', labelsize=fontsize-4)
    plt.tight_layout()
    plt.savefig('reports/figures/learning_rate_vs_accuracy.pdf', dpi=300, bbox_inches='tight')


def plot_final_run_curves():
    list_of_csvs = os.listdir("reports/training curves/final run")
    list_of_csvs = [x for x in list_of_csvs if x.endswith(".csv")]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
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
            
        v1_best = 68
        v2_best = 15
        
        best = [v1_best, v2_best]
        best_y_train = [training_losses['ConvNeXtV1'][v1_best-1], training_losses['ConvNeXtV2'][v2_best-1]]
        best_y_val = [validation_losses['ConvNeXtV1'][v1_best-1], validation_losses['ConvNeXtV2'][v2_best-1]]
        
        if best_y_train[0] < best_y_train[1]:
            text_offset_train = [0.000075, -0.000075]
        else:
            text_offset_train = [-0.000075, 0.000075]
        
        if best_y_val[0] < best_y_val[1]:
            text_offset_val = [0.003, -0.003]
        else:
            text_offset_val = [-0.003, 0.003]
        
        # Plot the training and validation losses
        colors = ['indianred', 'royalblue']
        x = df['Step']
        font_size = 20
        for i, model_name in enumerate(model_names):
            # long_model_name = 'ConvNeXtV1' if 'v1' in model_name else 'ConvNeXtV2'
            # plot training loss
            ax[0].plot(x, training_losses[model_name], label=f'{model_name} - train', color=colors[i])
            # plot validation loss
            ax[1].plot(x, validation_losses[model_name], label=f'{model_name} - val', color=colors[i], linestyle='dashed')
            
            # Plot point where best model was selected
            ax[0].scatter(best[i], best_y_train[i], color=colors[i], marker='.', s=100)
            ax[1].scatter(best[i], best_y_val[i], color=colors[i], marker='.', s=100)
            
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
            axis.legend(fontsize=font_size-8)
    
    fig.suptitle('Straw Level Models Final Training and Validation Losses', fontsize=font_size)
    
    plt.tight_layout()
    plt.savefig("reports/training curves/final run/straw_model_final_run.pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # analyse_model_selection()
    # plot_training_curves()
    # plot_hyper_parameters_vs_accuracy()
    plot_final_run_curves()