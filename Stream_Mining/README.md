# Stream Mining

Implementation of probabilistic and streaming algorithms for analysing large-scale data streams under memory constraints.

## Overview

This repository explores techniques for processing high-volume streaming data where storing all observations is impractical. The projects focus on efficient sampling, membership testing, and cardinality estimation.

## Methods

### Reservoir Sampling
- Implemented reservoir sampling for maintaining a representative sample from a continuous data stream.
- Ensured equal selection probability for all observations regardless of stream length.

### Bloom Filters
- Implemented Bloom filters using multiple hash functions.
- Evaluated membership queries and false positive behaviour for large collections of records.

### Flajolet-Martin Algorithm
- Implemented probabilistic cardinality estimation using hash-based trailing zero counts.
- Estimated the number of distinct elements in large data streams.

## Skills

Python, stream processing, probabilistic algorithms, hash functions, sampling methods, cardinality estimation, big data analytics
