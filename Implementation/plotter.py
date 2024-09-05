import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
def get_percent_na(df: pd.DataFrame):
    missing_percentage = {}

    for column_name, column in df.items():
        total_entries = len(column)
        missing_entries = column.isna().sum()
        percentage_missing = (missing_entries / total_entries) * 100
        missing_percentage[column_name] = percentage_missing

    return pd.DataFrame(list(missing_percentage.items()), columns=['', 'Percentage Missing/Invalid'])

'''
CUSTOM PLOTTING FUNCTION FOR PLOTTING NICE HISTOGRAMS
'''
def plot_histogram(series: pd.Series, title: str, ax=None, log_scale=False, bins='auto', rotation=0):
    """
    Plots a histogram for the given pandas Series. If an axis object is provided,
    it plots on that axis; otherwise, it creates its own figure.

    Parameters:
    - series (pd.Series): The series to plot.
    - ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new figure is created.
    - title (str): The title for the plot.
    - log_scale (bool): Whether to apply a logarithmic scale to the x-axis.
    - bins (int or sequence of scalars, optional): The number of bins or specific bin edges.
    - rotation (float): Rotation angle for x-axis labels.
    """
    if ax is None:
        # Create a new figure and axis if none are provided
        fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot the histogram
    sns.histplot(series, kde=True, ax=ax, bins=bins)
    ax.set_xlabel(series.name)
    ax.set_ylabel('Count')

    # Apply log scale if specified
    if log_scale:
        ax.set_xscale('log')
        ax.set_title(f'Histogram of {title} (log scale)')
    else:
        ax.set_title(f'Histogram of {title}')

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticks(), rotation=rotation)

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
