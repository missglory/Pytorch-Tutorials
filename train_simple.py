import torch

# torch.cuda.is_available()


import torch.nn as nn
import torch.optim as optim

class WeightedTimeseries(nn.Module):
    def __init__(self, num_columns):
        super(WeightedTimeseries, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_columns))

    def forward(self, timeseries_data):
        weighted_sum = torch.sum(timeseries_data * self.weights, dim=1)
        return weighted_sum

def optimize_weights(timeseries_data, num_iterations=1000):
    model = WeightedTimeseries(timeseries_data.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss = torch.sum(model(timeseries_data))
        loss.backward()
        optimizer.step()

    return model.weights.detach()

# Example usage:
if __name__ == '__main__':
    timeseries_data = torch.randn(100, 10)  # Replace with your actual data
    optimal_weights = optimize_weights(timeseries_data)
    print(optimal_weights)
