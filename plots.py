def plot_committor_values(committor_values, biases=None):

    if biases is None:
        biases = np.ones(len(committor_values))
    
    fig = plt.figure(figsize=(4,3))
    plt.plot(committor_values, ':', color='black')
    plt.scatter(np.arange(len(committor_values)), committor_values,
                s = biases / np.sum(biases) * 500,
                color='dodgerblue', alpha=.5, zorder=10, label='selection bias')
    plt.grid()
    plt.xlabel('Frame index')
    plt.ylabel('Committor value')
    plt.set_yticks(np.linspace(0, 1, 11))
    plt.legend()
    return fig

