import torch


def mse_loss(predicted_Y: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Define the Mean Squared Error (MSE) loss function.

    Parameters:
    predicted_Y (torch.Tensor): Predicted Y values.
    Y (torch.Tensor): Actual Y values.

    Returns:
    torch.Tensor: The MSE loss.
    """
    return 0.5 * torch.mean((predicted_Y - Y) ** 2, dim=1)