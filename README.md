This repository is a cloned version of [GraphRag](https://github.com/microsoft/graphrag), where the Leiden algorithm for community extraction from graphs has been modified.

## Repository Structure

In this repository, you can change the algorithm used for clustering graphs. You can select and replace each algorithm in the code. 

In the `graphrag/index/operations` directory, you will also find various implementations of the Leiden algorithm:

- **`leiden_changed.py`**: This file contains the Leiden algorithm with a BFS traversal over nodes.
- **`leiden_memo.py`**: This file implements the algorithm using memoization to remember the best node movements.
- **`leiden_with_threshold.py`**: This file includes an algorithm that uses a threshold to improve the modularity metric during node movement. If the movement only marginally improves the metric (according to the threshold), it will not be performed.


## Installation

1. Clone the repository:
   
bash
   git clone https://github.com/mashagodunova/graphrag.git
   cd yourrepository
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Replacing the Leiden Algorithm

To select the desired algorithm for extracting communities from the graph:

1. Navigate to the file:
   ```
   graphrag/index/operations/clustergraph.py
   

2. Find the import line that looks something like this:
   
python
   from graspologic.partition import hierarchical_leiden
   ```

3. Replace it with the desired algorithm. For example, to use `leidenwiththreshold`, change it to:
   ```python
   from graspologic.partition import leidenwiththreshold
   ```

4. Save the changes and run your code.

## Usage

After updating the algorithm, you can run your code for community extraction in the graph.
