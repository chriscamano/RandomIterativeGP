import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

def plot_gp_simple(train_x, train_y, test_x, GP_mean, std, test_y):
    """
    Plots a simple GP result.

    Parameters:
      train_x: training input data (should have same length as train_y)
      train_y: training targets
      test_x: test input data
      GP_mean: predictive mean at test inputs
      std: predictive standard deviation at test inputs
      test_y: true function values at test inputs
    """
    # Convert inputs to 1D numpy arrays.
    x_vals = np.array(test_x).ravel()
    GP_mean = np.array(GP_mean).ravel()
    std = np.array(std).ravel()
    train_x = np.array(train_x).ravel()
    train_y = np.array(train_y).ravel()
    test_y = np.array(test_y).ravel()

    plt.figure(figsize=(10, 5), dpi=400)
    
    plt.plot(x_vals, GP_mean, color='navy', lw=2, label=r'GPR mean $\overline{m}$')
    plt.fill_between(x_vals, GP_mean - 1.96*std, GP_mean + 1.96*std, 
                     color='navy', alpha=0.2, label='Uncertainty')
    
    plt.scatter(train_x, train_y, s=40, color='darkorange', 
                edgecolors='black', label='Noisy Observations', zorder=3)
    plt.plot(x_vals, test_y, color='green', alpha=0.6, 
             label="True Function", zorder=2)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, fancybox=True, shadow=False, framealpha=0.9)
    plt.show()

def plot_gp_sample(
    train_x, train_y, test_x_extended, test_y,
    true_function, GP_mean, GP_covariance, num_samples=45
):
    # GP_mean, GP_covariance = gpr.predict(test_x_extended)
    
    GP_mean = np.array(GP_mean).ravel()  
    GP_covariance = np.array(GP_covariance)
    std = np.sqrt(np.diag(GP_covariance))
    
    # Draw samples from the multivariate Gaussian.
    samples = np.random.multivariate_normal(GP_mean, GP_covariance, size=num_samples)
    
    plt.figure(figsize=(10, 5), dpi=400)
    colors = cm.Blues(np.linspace(0.3, 0.8, num_samples))
    x_vals = np.array(test_x_extended).ravel()
    for i in range(num_samples):
        plt.plot(x_vals, samples[i], color=colors[i], alpha=0.3, lw=1)
    
    uncertainty_handle = plt.fill_between(
        x_vals, GP_mean -1.96* std, GP_mean +1.96* std,
        color='royalblue', alpha=0.4, label='Uncertainty',
        linestyle="-", zorder=3,
    )
    
    mean_handle, = plt.plot(
        x_vals, GP_mean,
        color='darkblue', lw=2.5, label=r'GPR mean $\overline{m}$', zorder=4,
    )
    
    noisy_handle = plt.scatter(
        np.array(train_x).ravel(), np.array(train_y).ravel(),
        s=45, color='darkorange', edgecolors='black',
        label='Noisy Observations', zorder=5, linewidth=1
    )
    
    # Determine true function values over test_x_extended.
    if test_y.shape == test_x_extended.shape:
        true_y_extended = test_y
    else:
        true_y_extended = true_function(test_x_extended)
    
    true_handle, = plt.plot(
        x_vals, np.array(true_y_extended).ravel(),
        color='green', alpha=1, lw=2, label="True Function", zorder=3
    )
    
    # Draw dashed vertical lines indicating the training interval boundaries.
    plt.axvline(x=-3, color='black', linestyle='dashed', alpha=0.5, lw=1.8)
    plt.axvline(x=3, color='black', linestyle='dashed', alpha=0.5, lw=1.8)
    
    # Shade regions outside the training interval.
    plt.axvspan(-5, -3, color='gray', alpha=0.1)
    plt.axvspan(3, 5, color='gray', alpha=0.1)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    handles = [mean_handle, uncertainty_handle, noisy_handle, true_handle]
    plt.legend(handles=handles, fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, fancybox=True, shadow=True, framealpha=1)
    plt.show()

def plot_gp_simple_regions(train_x, train_y, test_x,test_y, GP_mean, lower, upper):
    """
    Plots a simple GP result.

    Parameters:
      train_x: training input data (should have same length as train_y)
      train_y: training targets
      test_x: test input data
      GP_mean: predictive mean at test inputs
      lower: lower confidence bound at test inputs
      upper: upper confidence bound at test inputs
      test_y: true function values at test inputs
    """
    # Convert inputs to 1D numpy arrays.
    x_vals = np.array(test_x).ravel()
    GP_mean = np.array(GP_mean).ravel()
    lower = np.array(lower).ravel()
    upper = np.array(upper).ravel()
    train_x = np.array(train_x).ravel()
    train_y = np.array(train_y).ravel()
    test_y = np.array(test_y).ravel()

    plt.figure(figsize=(10, 5), dpi=400)
    
    plt.plot(x_vals, GP_mean, color='navy', lw=2, label=r'GPR mean $\overline{m}$')
    plt.fill_between(x_vals, lower, upper, 
                     color='navy', alpha=0.2, label='Uncertainty')
    
    plt.scatter(train_x, train_y, s=40, color='darkorange', 
                edgecolors='black', label='Noisy Observations', zorder=3)
    plt.plot(x_vals, test_y, color='green', alpha=0.6, 
             label="True Function", zorder=2)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, fancybox=True, shadow=False, framealpha=0.9)
    plt.show()