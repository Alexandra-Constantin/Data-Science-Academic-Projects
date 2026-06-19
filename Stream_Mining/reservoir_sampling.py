# No external libraries are allowed to be imported in this file
import random

def mock_datastream():
    """This function is a mock datastream generator. It yields transactions one by one.
    It is used for testing the reservoir_sampling function. It is not allowed to change
    this function.

    Yields:
        transactions: A transaction from the datastream
    """
    for _ in range(10_000):
        yield random.gauss(10, 100) * (1 + 0.0005)

def reservoir_sampling(k, datastream):
    """This function should contain the code for the reservoir sampling algorithm.
    As an input it takes the sample size k and a datastream which is a generator 
    that yields the transactions one by one. Note that the resulting sample should be
    representative of the whole datastream.

    Args:
        k (int): The sample size
        datastream (func): The datastream generator that yields the transactions one by one

    Returns:
        list[transactions]: A list of size k containing the sampled transactions.
    """
    if k <= 0:
        return []

    sample = []
    for i, transaction in enumerate(datastream()):
        if i < k:
            # Fill the reservoir with the first k elements
            sample.append(transaction)
        else:
            u = random.random() # draw u ~ Unif(0,1)
            if u < (k / (i + 1)):
                # replace a uniformly random slot in the reservoir
                r = random.randrange(k)  # integer in {0, 1, ..., k-1}
                sample[r] = transaction
        
    return sample

if __name__ == "__main__":
    # You can use this main section for testing the reservoir_sampling function
    sample = reservoir_sampling(5000, mock_datastream)
    print(len(sample), sample[:5])
