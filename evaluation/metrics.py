"""Metrics for calculating accuracy of recommender system.

Used to evaluate and compare systems.
"""

import numpy as np


class Metrics:
    """Calculate the accuracy for recsys."""

    def MAE(predictions):
        """
        Calculate the mean absolute error.

        Parameters
        ----------
        predictions : list of predictions
            The list of generated predictions
                (userid, itemid, true result, estimated result, success?).
            Must have a true_result and estimate as third and fourth
            value in the set.

        Returns
        -------
        float
            The mean absolute error of predictions.

        """
        if not predictions:
            raise ValueError('Predictions list cannot be empty.')

        mae = np.mean([float(abs(true_result - est))
                       for(_, _, true_result, est, _) in predictions])

        return mae

    def RMSE(predictions):
        """
        Calculate the root mean squared error.

        Parameters
        ----------
        predictions : list of predictions
            The list of generated predictions.
                (userid, itemid, true result, estimated result, success?).
            Must have a true_result and estimate as third and fourth
            value in the set.

        Returns
        -------
        float
            The mean squared error of predictions.

        """
        if not predictions:
            raise ValueError('Predictions list cannot be empty.')

        rmse = np.square(
            np.mean([float(abs(true_result - est)**2)
                     for(_, _, true_result, est, _) in predictions]))

        return rmse

    def HitRate(topNPredictions, leftOutPredictions):
        """
        HR = #hits/#users.

        Parameters
        ----------
        topNPredictions : TYPE
            DESCRIPTION.
        leftOutPredictions : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError()
