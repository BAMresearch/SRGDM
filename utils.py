import matplotlib.pyplot as plt
from torch.utils import data


def plot_gdm(data, ax=None, vmin=None, vmax=None):
    if ax is None:
        fig, ax = plt.subplots();
        
    ax.axis("off");
    return ax.imshow(data.squeeze().detach().numpy(), vmin=vmin, vmax=vmax)


def plot_dict(input_dict, figsize, cbar_base_col=0):
    """ Plots the input of a dict. The dict should contain GDM data.
    
    Input:
    input_dict: ...
    figsize: ... (automatically in the future?)
    cbar_base_col: Which col should be used as baseline vmin and vmax of plt.imshow.
    """
    
    rows = len(next(iter(input_dict.values())))
    cols = len(input_dict)

    fig, axes = plt.subplots(rows, len(input_dict), constrained_layout=True, figsize=figsize)  
    
    # Loop over rows (amount of samples)
    for row in range(rows):
        vmin=None
        vmax=None

        vmin = input_dict[list(input_dict.keys())[cbar_base_col]][row].min()
        vmax = input_dict[list(input_dict.keys())[cbar_base_col]][row].max()   
        
        # Loop over cols (amount of models)
        for col, i in zip(input_dict, range(cols)):
            try:
                # Try for multiple rows
                plot_gdm(input_dict[col][row].squeeze(), axes[row][i], vmin=vmin, vmax=vmax);
                if row == 0:
                    axes[row][i].set_title(col)
        
            except:
                # otherwise only single row
                plot_gdm(input_dict[col][row].squeeze(), axes[i], vmin=vmin, vmax=vmax);
                if row == 0:
                    axes[i].set_title(key)
    
    return fig, axes


def draw_random_samples(n_samples, dataset, sequential=False):
    """ Take a dataset and draw n random samples from the dataset. 
    
    Input:    
    n_samples: The amount of samples
    dataset: torch.data.Dataset
    sequential: True, if the sequence length dimension should be removed
    
    Output:
    X_list: A list containing n tensors with input data for model
    y_list: A list containing n tensors with true gas distribution data
    """
    
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    data_iter = iter(loader) 
    
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        X, y = data_iter.next()
        X = X.squeeze(1) # remove the sequence length dimension
        X_list.append(X)
        y_list.append(y)
    
    return X_list, y_list
