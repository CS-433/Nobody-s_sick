"""function for plot."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_histograms(data_list, features_list, num_cols=4):
    """
    Creates histograms for the given data.

    Parameters:
    - data_list: List of datasets to plot.
    - num_bins: Number of bins for the histogram (default: 10).
    - num_cols: Number of columns in the subplot grid (default: 2).
    """
    num_plots = len(data_list)  # Total number of datasets
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division for rows

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Define a list of colors for the histograms
    colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

    # Generate histograms for each dataset
    for i, data in enumerate(data_list):

        # Drop NaNs
        # data = data[~np.isnan(data)]
        
        # Get unique values and counts
        unique_values, counts = np.unique(data, return_counts=True)
        
        # Plot the histogram with unique values
        axs[i].bar(unique_values, counts, color=colors[i], edgecolor='black', alpha=0.7)
        axs[i].set_title(features_list[i])  
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xticks(unique_values)

    # Hide any unused subplots if num_plots < num_rows * num_cols
    for j in range(num_plots, num_rows * num_cols):
        axs[j].axis('off')  # Turn off the axis for unused subplots

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_boxplots(data_list, features_list, num_cols=4):
    """
    Creates histograms for the given data.

    Parameters:
    - data_list: List of datasets to plot.
    - num_bins: Number of bins for the histogram (default: 10).
    - num_cols: Number of columns in the subplot grid (default: 2).
    """
    num_plots = len(data_list)  # Total number of datasets
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division for rows

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Define a list of colors for the histograms
    colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

    # Generate histograms for each dataset
    for i, data in enumerate(data_list):
        
        # Plot the histogram with unique values
        sns.violinplot(ax=axs[i], data=data)
        axs[i].set_title(features_list[i])  
        axs[i].set_ylabel('Value')
        axs[i].set_xlabel('')

    # Hide any unused subplots if num_plots < num_rows * num_cols
    for j in range(num_plots, num_rows * num_cols):
        axs[j].axis('off')  # Turn off the axis for unused subplots

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_loss_variable_gamma(losses_var_gamma, param):
    
    ax = plt.axes()
    
    #Loop on the different values of gamma
    for idx, losses in enumerate(losses_var_gamma):
        ax.semilogy(losses, label = "gamma = %.1e" % param[idx])
    
    plt.title("Loss for different learning rates")
    plt.xlabel("# iterations")
    plt.ylabel("Loss")
    
    plt.legend()
    plt.show()

def plot_train_test(ax, title, train_errors, test_errors, params, param_name):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    ax.semilogx(params, train_errors, color="b", marker="*", label="Train error")
    ax.semilogx(params, test_errors, color="r", marker="*", label="Test error")
    ax.set_xlabel(param_name)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    leg = ax.legend(shadow=True)
    leg.set_frame_on(False)
