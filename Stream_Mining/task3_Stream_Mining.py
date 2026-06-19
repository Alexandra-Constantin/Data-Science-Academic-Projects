import hashlib
import statistics


class FlajoletMartin:
    def __init__(self, num_hashes=10):
        self.max_trailing_zeros = [0] * num_hashes

    def hash_function(self, item, seed):
        """Hash the given item using SHA-256 and return an integer."""
        
        # Combine element and seed into a string
        item_seeded = f"{seed}-{item}"  # like "0-element-1"
        # SHA-256
        hash_object = hashlib.sha256(item_seeded.encode())
        # Convert a hexadecimal hash value to an integer
        hash_value = int(hash_object.hexdigest(), 16)
        return hash_value


    def count_trailing_zeros(self, x):
        """Count the number of trailing zeros in the binary representation of a number."""
        
        trailing_zeros_count = 0
        if x == 0:
            return 256  # if all bits are zero
        while (x & 1) == 0:
            trailing_zeros_count += 1
            x >>= 1

        return trailing_zeros_count


    def add(self, item):
        """Add a new item to the estimator and update the maximum trailing zeros."""
        
        for i in range(len(self.max_trailing_zeros)):
            h = self.hash_function(item, seed=i) # compute the hash value
            r = self.count_trailing_zeros(h) # count number of trailing zeros in a hash value
            if r > self.max_trailing_zeros[i]: # update maximum trailing zero value of current hash function
                self.max_trailing_zeros[i] = r
            

    def estimate_number(self):
        """Estimate the number of distinct elements by taking the median from multiple hashes."""
        
        # Calculate the median of the maximum number of trailing zeros of multiple hash functions
        median_trailing_zeros = statistics.median(self.max_trailing_zeros)
        
        # Estimate the cardinality from the median R
        estimated_number = 2 ** median_trailing_zeros / 0.75351
        return estimated_number, median_trailing_zeros


# Inspection Results
if __name__ == "__main__":
    fm = FlajoletMartin(num_hashes=20)
    unique_elements = [f"element-{i}" for i in range(100000)]  # Simulating 100 000 unique elements

    for element in unique_elements:
        fm.add(element)

    estimate, median_R = fm.estimate_number()
    print(f"Median R: {median_R}")
    print(f"Estimated Cardinality: {estimate}")
