import numpy as np
import torch


class calcualte_force:
    def __init__(
        self,
    ):
        """
        Initialize regression model parameters and device for computation.
        """
        self.curved_object_intercept = 22.26361025
        self.curved_object_slope = -0.17844147
        self.flat_object_intercept = 7.66228198
        self.flat_object_slope = -0.05622074

        self.device = "cuda:0"

    def regression(self, suction_score):
        """
        Compute force based on the provided suction deformation score using a regression model.
        
        Parameters:
        - suction_score: Score indicating quality of suction (float).
        
        Returns:
        - force: Predicted force (torch.Tensor).
        """
        self.suction_score = suction_score * 100
        if self.suction_score < 80:
            prediction = (
                self.suction_score * self.curved_object_slope
                + self.curved_object_intercept
            )
            mu, sigma = prediction, np.sqrt(1.6678003887167776)
            force = torch.normal(mu, sigma).to(self.device)
        else:
            prediction = (
                self.suction_score * self.flat_object_slope + self.flat_object_intercept
            )
            mu, sigma = prediction, np.sqrt(1.4155587192333443)
            force = torch.normal(mu, sigma).to(self.device)
        return force
