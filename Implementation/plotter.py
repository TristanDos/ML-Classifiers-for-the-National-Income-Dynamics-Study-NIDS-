import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

'''
CUSTOM PLOTTING FUNCTION FOR PLOTTING NICE BAR GRAPHS
'''
def plot_bar(series: pd.Series, title: str, ax=None, log_scale=False, rotation=0):
    """
    Plots a bar graph for the given pandas Series. If an axis object is provided,
    it plots on that axis; otherwise, it creates its own figure.

    Parameters:
    - series (pd.Series): The series to plot. Typically, this should contain categorical data or discrete counts.
    - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
    - title (str): The title for the plot.
    - log_scale (bool): Whether to apply a logarithmic scale to the y-axis.
    """
    if ax is None:
        # Create a new figure and axis if none are provided
        fig, ax = plt.subplots(figsize=(7, 4))
    
    value_counts = series.value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    ax.set_xlabel(series.name)
    ax.set_ylabel('Count')

    ax.set_xticklabels(value_counts.index, rotation=rotation)
    if log_scale:
        ax.set_yscale('log')
        ax.set_title(f'Bar Graph of {title} (log scale)')
    else:
        ax.set_title(f'Bar Graph of {title}')
    
    if ax is None:
        plt.tight_layout()
        plt.show()

'''
USED FOR OBTAINING THE PERCENTAGE NAN ENTRIES OF A COLUMN
'''
def get_percent_na(col: list):
    missing_count = 0
    total_entries = len(col)

    for i in col:
        if i is pd.NA: missing_count += 1

    percentage_missing = (missing_count / total_entries) * 100

    return float(percentage_missing)

'''
CUSTOM PLOTTING FUNCTION FOR PLOTTING NICE HISTOGRAMS
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(series: pd.Series, title: str, ax=None, log_scale=False, bins='auto', rotation=0):
    """
    Plots a histogram for continuous data or a count plot for categorical data.
    If an axis object is provided, it plots on that axis; otherwise, it creates its own figure.

    Parameters:
    - series (pd.Series): The series to plot.
    - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
    - title (str): The title for the plot.
    - log_scale (bool): Whether to apply a logarithmic scale to the x-axis (only for continuous data).
    - bins (int or sequence of scalars, optional): The number of bins or specific bin edges (for continuous data).
    - rotation (float): Rotation angle for x-axis labels.
    """
    if ax is None:
        # Create a new figure and axis if none are provided
        fig, ax = plt.subplots(figsize=(7, 4))
    
    # Check if the data is categorical
    if pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
        # Plot a countplot for categorical data
        sns.countplot(x=series, ax=ax)
        ax.set_xlabel(series.name)
        ax.set_ylabel('Count')
        ax.set_title(f'Binned Histogram of {title}')
    else:
        # Plot the histogram for continuous data
        sns.histplot(series, kde=True, ax=ax, bins=bins)
        ax.set_xlabel(series.name)
        ax.set_ylabel('Count')

        # Apply log scale if specified (only for continuous data)
        if log_scale:
            ax.set_xscale('log')
            ax.set_title(f'Histogram of {title} (log scale)')
        else:
            ax.set_title(f'Histogram of {title}')

    # Rotate x-axis labels (useful for categorical data)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    # Display the plot if no axis was provided
    if ax is None:
        plt.tight_layout()
        plt.show()


'''
CUSTOM PLOTTING FUNCTION FOR PLOTTING NICE BOX AND WHISKER PLOTS
'''
def plot_boxplot(series: pd.Series, title: str, ax=None, log_scale=False):
    """
    Plots a box plot for the given pandas Series on the provided axis.

    Parameters:
    - series (pd.Series): The series to plot.
    - ax (matplotlib.axes.Axes): The axis to plot on.
    - title (str): The title for the plot.
    - log_scale (bool): Whether to apply a logarithmic scale to the x-axis.
    """
    if ax is None:
        # Create a new figure and axis if none are provided
        fig, ax = plt.subplots(figsize=(7, 4))
    
    sns.boxplot(x=series, ax=ax)
    ax.set_xlabel(series.name)
    if log_scale:
        ax.set_xscale('log')
        ax.set_title(f'Box Plot of {title} (log scale)')
    else:
        ax.set_title(f'Box Plot of {title}')
    ax.figure.tight_layout()

    if ax is None:
        plt.tight_layout()
        plt.show()

'''
CUSTOM PLOTTING FUNCTION FOR PLOTTING COMPARISON BARGRAPH
'''
def plot_model_comparison(df: pd.DataFrame, title: str, ax=None, log_scale=False, rotation=0, group_by_model=False):
    """
    Plots a grouped bar graph comparing scores across different models or metrics.

    Parameters:
    - df (pd.DataFrame): A DataFrame where each column represents a model and each row represents a metric/category.
    - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
    - title (str): The title for the plot.
    - log_scale (bool): Whether to apply a logarithmic scale to the y-axis.
    - rotation (int): Rotation for x-tick labels.
    - group_by_model (bool): If True, group by models on the x-axis; otherwise, group by metrics.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size if needed

    if group_by_model:
        # Reshape data to have models on the x-axis and metrics as hue
        df_long = df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
        sns.barplot(x='Metric', y='Score', hue='index', data=df_long, ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
    else:
        # Default: Group by metrics on the x-axis
        df_long = df.reset_index().melt(id_vars='index', var_name='Model', value_name='Score')
        sns.barplot(x='index', y='Score', hue='Model', data=df_long, ax=ax)
        ax.set_xlabel('Metric')

    ax.set_ylabel('Score')
    ax.set_xticklabels(df.columns if group_by_model else df.index, rotation=rotation)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_title(f'Comparison of {title} (log scale)')
    else:
        ax.set_title(f'Comparison of {title}')
    
    ax.legend(title='Metric' if group_by_model else 'Model', loc="lower right", fancybox=True, framealpha=0.9)

    if ax is None:
        plt.tight_layout()
        plt.show()

'''
CUSTOM PLOTTING FUNCTION FOR PLOTTING COMPARISON HEATMAPS
'''
def plot_confusion_matrix_heatmaps(df: pd.DataFrame, title: str, ax=None, cmap='Oranges', solo=True):
    """
    Plots heatmaps comparing confusion matrices from different models.

    Parameters:
    - df (pd.DataFrame): A DataFrame where each column represents the confusion matrix of a model,
                         and each row represents a class/category.
    - title (str): The title for the plot.
    - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
    - cmap (str): Colormap to use for the heatmap.
    """
    num_models = df.shape[1]
    num_classes = df.shape[0]
    
    if not solo:
        # Create a grid of subplots for each model's confusion matrix
        grid = plt.GridSpec(1, num_models, wspace=0.3)

        for i, model in enumerate(df.columns):
            ax_model = plt.subplot(grid[0, i])  # Create a subplot for each model
            sns.heatmap(df[model].values.reshape(int(np.sqrt(num_classes)), int(np.sqrt(num_classes))), 
                        annot=True, fmt='d', cmap=cmap, 
                        ax=ax_model, cbar=i == 0)  # Show colorbar only for the first subplot
            ax_model.set_title(model)
            ax_model.set_xlabel('Predicted')
            ax_model.set_ylabel('True')

        plt.suptitle(f'Comparison of Confusion Matrices - {title}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
        plt.show()

    else:
        fig, ax = plt.subplots(nrows=num_models, ncols=1, figsize=(30, 40))
        plt.subplots_adjust(hspace=0.4)
        fig.patch.set_facecolor('#eeeee1')

        for i, model in enumerate(df.columns):
            sns.heatmap(df[model].values.reshape(int(np.sqrt(num_classes)), int(np.sqrt(num_classes))), 
                        annot=True, fmt='d', cmap=cmap, 
                        ax=ax[i], annot_kws={"size": 50})  # Show colorbar only for the first subplot
            ax[i].set_title(model, fontsize=80)
            ax[i].set_xlabel('Predicted', fontsize=50)
            ax[i].set_ylabel('Actual', fontsize=50)

        # plt.title(title)
        plt.show()
        plt.close()


def save_model(model, model_path: str):
    """Saves a model to the model path.

    Parameters
    ----------
    model : any
        Model in any format
    model_path : str
        Path to save model file
    """    

    # save
    with open(model_path,'wb') as f:
        pickle.dump(model,f)

def load_model(model_path: str):
    """Loads a model from specified model path.

    Parameters
    ----------
    model_path : str
        Path to load model from

    Returns
    -------
    any
        Model reconstructed from pkl file
    """    

    # load
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model