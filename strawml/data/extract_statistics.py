import numpy as np



def update(existing_aggregate: tuple[int, np.array, np.mean], new_value: np.array) -> tuple[int, np.array, np.array]:
    """ Welford's Online Algorithm, "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    For a new value new_value, compute the new count, new mean, the new M2.
    count aggregates the number of samples seen so far
    mean accumulates the mean of the entire dataset
    M2 aggregates the squared distance from the mean
    
    Parameters:
    ----------
        existing_aggregate (tuple[int, np.array, np.array]): Tuple containing the existing count, mean, and M2
        new_value (np.array): The new value to update the aggregate with
    
    Returns:
    -------
        tuple[int, np.array, np.array]
        A tuple containing the following elements:
        - count (int): The new count
        - mean (np.array): The new mean
        - M2 (np.array): The new M2
    """
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)


def finalize(existing_aggregate: tuple[int, np.array, np.array]) -> tuple[np.array, np.array, np.array]:
    """ Welford's Online Algorithm, "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
    Retrieve the mean, variance and sample variance from an aggregate

    Parameters:
    ----------
        existing_aggregate (tuple[int, np.array, np.array]): Tuple containing the existing count, mean, and M2
    
    Returns:
    -------
        tuple[int, np.array, np.array]
        A tuple containing the following elements:
        - mean (np.array): The mean of the data
        - variance (np.array): The variance of the data
        - sample_variance (np.array): The sample variance of the data
    """
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sample_variance)
