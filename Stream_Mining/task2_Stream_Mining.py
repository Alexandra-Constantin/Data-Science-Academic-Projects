# Set up a fake data set with legal bank account numbers and a fake data set with illegal bank account numbers.
from hashlib import sha256
import numpy as np
import functools


def create_hash_functions(num_hash_functions, size_bit_array):
    """Create a list of runnable hash functions.
    It is important to use lambda functions to create the hash functions.

    Args:
        num_hash_functions (Int): The number of hash functions to create.
        size_bit_array (Int): The size of the bit array.

    Returns:
        list[lambda]: A list containing the hash functions.
    """
    # Generate a list of hash functions
    hash_functions = []
    
    for i in range(num_hash_functions):
        # each lambda includes its index i in the input to make make different hash functions
        h = (lambda x, i=i, m=size_bit_array:
            int(sha256(f"{i}|{x}".encode("utf-8")).hexdigest(), 16) % m)
        hash_functions.append(h)

    return hash_functions


def add_to_bloom_filter(bloom_filter, hash_functions, bank_account):
    """Set the k bits for this bank_account to 1."""
    
    for h in hash_functions:
        idx = h(bank_account) # for each hash function, get the index
        bloom_filter[idx] = 1 # flip the switch to 1
    return bloom_filter


def check_bloom_filter(bloom_filter, hash_functions, bank_account):
    """Return True iff all k hashed positions are 1 (i.e., 'possibly present')."""
    
    for h in hash_functions:
        idx = h(bank_account)
        if bloom_filter[idx] == 0:
            return False  # definitely not present
    return True  # possibly present (could be a false positive)

if __name__ == "__main__":
    # This section can be used to debug your submission

    nr_bank_accounts = 100_000

    # Create a list of legal bank account numbers
    real_bank_accounts = ["real" + str(i) for i in range(nr_bank_accounts)]

    # Set up the Bloom filter as an array 8 times as big as the number of bank accounts (per slides)
    bloom_filter = [0] * (8 * nr_bank_accounts)

    # Experiment with 2 hash functions (you can try raising it up to ~30 as suggested)
    hash_functions = create_hash_functions(2, 8 * nr_bank_accounts)

    # Enter all valid account numbers
    for account in real_bank_accounts:
        add_to_bloom_filter(bloom_filter, hash_functions, account)

    # Calculate the false positive rate
    fake_bank_accounts = ["fake" + str(i) for i in range(nr_bank_accounts)]
    false_positives = 0
    for fake_account in fake_bank_accounts:
        if check_bloom_filter(bloom_filter, hash_functions, fake_account):
            false_positives += 1
    print(f"False positive rate: {false_positives/nr_bank_accounts}")

    print("Fraction of bits set: ", np.sum(bloom_filter) / (nr_bank_accounts * 8))

    print("Is real12345 a valid account number?", check_bloom_filter(bloom_filter, hash_functions, "real12345"))
    print("Is real123456 a valid account number?", check_bloom_filter(bloom_filter, hash_functions, "real123456"))
    print("Is 12345 a valid account number?", check_bloom_filter(bloom_filter, hash_functions, "12345"))
