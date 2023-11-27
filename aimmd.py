"""
Auxiliary files for AIMMD.
"""


def shoot(x, y, inA, inB, engine, nsteps=100, max_length=10000, D=1.0, dt=1e-5):
    """
    Perform 2-way shooting from the shooting point (x, y).
    Stop upon reaching one of the two states.
  
    Parameters
    ----------
    x, y: float, shooting point position
    inA, inB: functions, check whether a point is in A or in B
    engine: to evolve x and y positions
    max_length: int, maximum length of a backward/forward subtrajectory
    nsteps: int, integration steps between subsequent frames
    D: float, diffusion coefficient
    dt: float, integration step
  
    Returns
    -------
    trajectory: (n, 2)-shaped numpy array
    shooting_index: position of shooting point in generated trajectory
    """
  
    # initialize backward subtrajectory at the shooting point
    backward_trajectory = np.zeros((max_length, 2)) + [x, y]
  
    # simulate until reaching one of the states
    for i in range(1, max_length):
        backward_trajectory[i] = engine(*backward_trajectory[i - 1])
        if (inA(backward_trajectory[i, None])[0] or
            inB(backward_trajectory[i, None])[0]):
            break
  
    # extract the segment in the transition region and flip directions
    backward_trajectory = backward_trajectory[:i + 1][::-1]
  
    # initialize backward subtrajectory at the shooting point
    forward_trajectory = np.zeros((max_length, 2)) + [x, y]
  
    # simulate until reaching one of the states
    for i in range(1, max_length):
        forward_trajectory[i] = engine(*forward_trajectory[i - 1])
        if (inA(forward_trajectory[i, None])[0] or
            inB(forward_trajectory[i, None])[0]):
            break
  
    # extract the segment in the transition region without the shooting point
    forward_trajectory = forward_trajectory[1:i+1]
  
    # join and return
    trajectory = np.append(backward_trajectory, forward_trajectory, axis=0)
    return trajectory, len(backward_trajectory) - 1

class Network(torch.nn.Module):
    """
    Pytorch neural network.
    Input features size: 2 (x, y).
    Output size: 1 (logit of the committor).
    """
    def __init__(self):
        super().__init__()
        self.call_kwargs = {}
        # layers & activations
        n = 512
        self.input = torch.nn.Linear(2, n)
        self.layer1 = torch.nn.Linear(n, n)
        self.layer2 = torch.nn.Linear(n, n)
        self.output = torch.nn.Linear(n, 1)
        self.activation1 = torch.nn.PReLU(n)
        self.activation2 = torch.nn.PReLU(n)
        self.reset_parameters()
    def forward(self, x):
        x = self.input(x)
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.output(x)
        return x
    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.output.reset_parameters()


def train(model, shooting_points, shooting_results, lr=1e-4):
    """

    """
    batch_size = 4096
    base_epochs = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.reset_parameters()
    model.train()

    # create training set
    descriptors = torch.tensor(shooting_points, dtype=torch.float)
    results = torch.tensor(shooting_results)
    norms = torch.tensor(np.sum(shooting_results, axis=1))

    # training weights
    weights = np.zeros(len(descriptors))
    A_to_A = shooting_results[:, 0] == 2
    weights[A_to_A] = 1 / np.sum(A_to_A)
    B_to_B = shooting_results[:, 1] == 2
    weights[B_to_B] = 1 / np.sum(B_to_B)
    A_to_B = (shooting_results[:, 0] == 1) * (shooting_results[:, 1] == 1)
    weights[A_to_B] = 1 / np.sum(A_to_B)
    weights /= np.sum(weights)

    # train model
    losses = []
    epochs = base_epochs # int(len(descriptors) ** 1/3 * base_epochs)
    for i in range(epochs):

        # train cycle
        def closure():
            optimizer.zero_grad()

            # training epoch
            q = model(descriptors)
            exp_pos_q = torch.exp(+q[:, 0])
            exp_neg_q = torch.exp(-q[:, 0])
            toA_contribution = results[:, 0] * torch.log(1. + exp_pos_q)
            toB_contribution = results[:, 1] * torch.log(1. + exp_neg_q)
            loss = torch.sum((toA_contribution + toB_contribution) / norms)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        losses.append(float(loss) / len(descriptors))

    model.eval()

    return model, losses


def evaluate(model, descriptors, batch_size=4096):
  """
  Model on descriptors.
  """
  values = []
  for batch in np.array_split(descriptors,
      max(len(descriptors) // batch_size, 1)):
    batch_values = model(torch.tensor(batch, dtype=torch.float))
    batch_values = batch_values.cpu().detach().numpy().ravel()
    batch_values = scipy.special.expit(batch_values)
    values.append(batch_values)
  return np.concatenate(values)


def selection_biases(committor_values, adaptation_bins=np.linspace(0, 1, 11)):
    """
    Density correction.
    """
  
    n_adaption_bins = len(adaptation_bins) - 1
    bin_weights = np.ones(n_adaption_bins)
    populations, _ = np.histogram(committor_values[1:-1], bins=adaptation_bins)
  
    # distribute selection biases of empty bins
    for _ in range(10):  # max recursive length
          for empty_bin_index in np.where(populations == 0)[0]:
            if empty_bin_index > 0 and empty_bin_index < n_adaption_bins - 1:
                bin_weights[empty_bin_index - 1] += bin_weights[empty_bin_index] / 2
                bin_weights[empty_bin_index + 1] += bin_weights[empty_bin_index] / 2
            elif empty_bin_index == 0:  # first bin empty
                bin_weights[empty_bin_index + 1] += bin_weights[empty_bin_index]
            else:  # last bin empty
                bin_weights[empty_bin_index - 1] += bin_weights[empty_bin_index]
            bin_weights[empty_bin_index] = 0.
  
    # compute selection biases
    selection_biases = np.zeros(len(committor_values))
    bin_indices = np.digitize(committor_values, adaptation_bins[1:-1])
    bin_indices[[0, -1]] = -1  # exclude
    for i in range(n_adaption_bins):
          selection_biases[bin_indices == i] = (
              bin_weights[i] / np.sum(bin_indices == i))
    selection_biases /= np.sum(selection_biases)
  
    return selection_biases
