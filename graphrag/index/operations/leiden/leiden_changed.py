import warnings
from typing import Any, NamedTuple, Optional, Union
from typing import List
import graspologic_native as gn
import networkx as nx
import numpy as np
import scipy
from beartype import beartype
from collections import deque

from graspologic.types import AdjacencyMatrix, Dict, GraphRepresentation, List, Tuple
from .. import utils
from ..preconditions import check_argument


class _IdentityMapper:
    def __init__(self) -> None:
        self._inner_mapping: Dict[str, Any] = {}

    def __call__(self, value: Any) -> str:
        as_str = str(value)
        mapped = self._inner_mapping.get(as_str, value)
        if mapped != value:
            raise ValueError(
                "str(value) results in a collision between distinct values"
            )
        self._inner_mapping[as_str] = mapped
        return as_str

    def original(self, as_str: str) -> Any:
        return self._inner_mapping[as_str]

    def __len__(self) -> int:
        return len(self._inner_mapping)


@beartype
def _bfs_connected_components(graph: nx.Graph) -> List[List[Any]]:
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            component = []
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    for neighbor in graph.neighbors(current):
                        if neighbor not in visited:
                            queue.append(neighbor)
            components.append(component)

    return components

@beartype
def _process_component(
    component: List[Any],
    graph: nx.Graph,
    identifier: _IdentityMapper,
    is_weighted: Optional[bool],
    weight_attribute: str,
    weight_default: float,
) -> Tuple[int, List[Tuple[str, str, float]]]:
    native_safe: List[Tuple[str, str, float]] = []
    for source in component:
        source_str = identifier(source)
        for target in graph.neighbors(source):
            if source <= target:  # To ensure no duplicate edges
                target_str = identifier(target)
                weight = graph[source][target].get(weight_attribute, weight_default)
                if not is_weighted and weight == 1:
                    weight = weight_default
                native_safe.append((source_str, target_str, float(weight)))
    return len(component), native_safe

@beartype
def _nx_to_edge_list(
    graph: nx.Graph,
    identifier: _IdentityMapper,
    is_weighted: Optional[bool],
    weight_attribute: str,
    weight_default: float,
) -> Tuple[int, List[Tuple[str, str, float]]]:
    check_argument(
        isinstance(graph, nx.Graph)
        and not (graph.is_directed() or graph.is_multigraph()),
        "Only undirected non-multi-graph networkx graphs are supported",
    )
    native_safe: List[Tuple[str, str, float]] = []
    edge_iter = (
        graph.edges(data=weight_attribute)
        if is_weighted is True
        else graph.edges(data=weight_attribute, default=weight_default)
    )
    for source, target, weight in edge_iter:
        source_str = identifier(source)
        target_str = identifier(target)
        native_safe.append((source_str, target_str, float(weight)))
    return graph.number_of_nodes(), native_safe


@beartype
def _adjacency_matrix_to_edge_list(
    matrix: AdjacencyMatrix,
    identifier: _IdentityMapper,
    check_directed: Optional[bool],
    is_weighted: Optional[bool],
    weight_default: float,
) -> Tuple[int, List[Tuple[str, str, float]]]:
    check_argument(
        check_directed is True and utils.is_almost_symmetric(matrix),
        "leiden only supports undirected graphs and the adjacency matrix provided "
        "was found to be directed",
    )
    shape = matrix.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            "graphs of type np.ndarray or csr.sparse.csr.csr_array should be "
            "adjacency matrices with n x n shape"
        )

    if is_weighted is None:
        is_weighted = not utils.is_unweighted(matrix)

    native_safe: List[Tuple[str, str, float]] = []
    if isinstance(matrix, np.ndarray):
        for i in range(0, shape[0]):
            source = identifier(i)
            for j in range(i, shape[1]):
                target = identifier(j)
                weight = matrix[i][j]
                if weight != 0:
                    if not is_weighted and weight == 1:
                        weight = weight_default
                    native_safe.append((source, target, float(weight)))
    else:
        rows, columns = matrix.nonzero()
        for i in range(0, len(rows)):
            source = rows[i]
            source_str = identifier(source)
            target = columns[i]
            target_str = identifier(target)
            weight = float(matrix[source, target])
            if source <= target:
                native_safe.append((source_str, target_str, weight))

    return shape[0], native_safe


@beartype
def _edge_list_to_edge_list(
    edges: List[Tuple[Any, Any, Union[int, float]]], identifier: _IdentityMapper
) -> Tuple[int, List[Tuple[str, str, float]]]:
    native_safe: List[Tuple[str, str, float]] = []
    temp_node_set = set()

    for source, target, weight in edges:
        source_str = identifier(source)
        target_str = identifier(target)
        weight_as_float = float(weight)
        if source_str != target_str:
            native_safe.append((source_str, target_str, weight_as_float))
            temp_node_set.add(source_str)
            temp_node_set.add(target_str)
    return len(temp_node_set), native_safe


@beartype
def _community_python_to_native(
    starting_communities: Optional[Dict[Any, int]], identity: _IdentityMapper
) -> Optional[Dict[str, int]]:
    if starting_communities is None:
        return None
    native_safe: Dict[str, int] = {}
    for node_id, partition in starting_communities.items():
        node_id_as_str = identity(node_id)
        native_safe[node_id_as_str] = partition
    return native_safe

@beartype
def _community_native_to_python(
    communities: Dict[str, int], identity: _IdentityMapper
) -> Dict[Any, int]:
    return {
        identity.original(node_id_as_str): partition
        for node_id_as_str, partition in communities.items()
    }


@beartype
def _validate_common_arguments(
    extra_forced_iterations: int = 0,
    resolution: Union[float, int] = 1.0,
    randomness: Union[float, int] = 0.001,
    random_seed: Optional[int] = None,
) -> None:
    check_argument(
        extra_forced_iterations >= 0,
        "extra_forced_iterations must be a non negative integer",
    )
    check_argument(resolution > 0, "resolution must be a positive float")
    check_argument(randomness > 0, "randomness must be a positive float")
    check_argument(
        random_seed is None or random_seed > 0,
        "random_seed must be a positive integer (the native PRNG implementation is"
        " an unsigned 64 bit integer)",
    )


@beartype
def leiden(
    graph: nx.Graph,
    starting_communities: Optional[Dict[Any, int]] = None,
    extra_forced_iterations: int = 0,
    resolution: Union[int, float] = 1.0,
    randomness: Union[int, float] = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
    is_weighted: Optional[bool] = None,
    weight_default: Union[int, float] = 1.0,
    check_directed: bool = True,
    trials: int = 1,
) -> Dict[Any, int]:
    _validate_common_arguments(
        extra_forced_iterations,
        resolution,
        randomness,
        random_seed,
    )
    check_argument(trials >= 1, "Trials must be a positive integer")

    identifier = _IdentityMapper()
    node_count: int
    edges: List[Tuple[str, str, float]]
    
    components = _bfs_connected_components(graph)
    native_safe_edges = []
    for component in components:
        _, component_edges = _process_component(
            component, graph, identifier, is_weighted, weight_attribute, weight_default
        )
        native_safe_edges.extend(component_edges)

    native_friendly_communities = _community_python_to_native(
        starting_communities, identifier
    )

    _quality, native_partitions = gn.leiden(
        edges=native_safe_edges,
        starting_communities=native_friendly_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=extra_forced_iterations + 1,
        use_modularity=use_modularity,
        seed=random_seed,
        trials=trials,
    )

    proper_partitions = _community_native_to_python(native_partitions, identifier)

    if len(proper_partitions) < graph.number_of_nodes():
        warnings.warn(
            "Leiden partitions do not contain all nodes from the input graph because input graph "
            "contained isolate nodes."
        )

    return proper_partitions


class HierarchicalCluster(NamedTuple):
    node: Any
    """Node id"""
    cluster: int
    """Leiden cluster id"""
    parent_cluster: Optional[int]
    """Only used when level != 0, but will indicate the previous cluster id that this node was in"""
    level: int
    """
    Each time a community has a higher population than we would like, we create a subnetwork
    of that community and process it again to break it into smaller chunks. Each time we
    detect this, the level increases by 1
    """
    is_final_cluster: bool
    """
    Whether this is the terminal cluster in the hierarchical leiden process or not
    """


class HierarchicalClusters(List[HierarchicalCluster]):
    """
    HierarchicalClusters is a subclass of Python's :class:`list` class with two
    helper methods for retrieving dictionary views of the first and final
    level of hierarchical clustering in dictionary form.  The rest of the
    HierarchicalCluster entries in this list can be seen as a transition
    state log of our :func:`graspologic.partition.hierarchical_leiden` process
    as it continuously tries to break down communities over a certain size,
    with the two helper methods on this list providing you the starting point
    community map and ending point community map.
    """

    def first_level_hierarchical_clustering(self) -> Dict[Any, int]:
        """
        Returns
        -------
        Dict[Any, int]
            The initial leiden algorithm clustering results as a dictionary
            of node id to community id.
        """
        return {entry.node: entry.cluster for entry in self if entry.level == 0}

    def final_level_hierarchical_clustering(self) -> Dict[Any, int]:
        """
        Returns
        -------
        Dict[Any, int]
            The last leiden algorithm clustering results as a dictionary
            of node id to community id.
        """
        return {entry.node: entry.cluster for entry in self if entry.is_final_cluster}


def _from_native(
    native_cluster: gn.HierarchicalCluster,
    identifier: _IdentityMapper,
) -> HierarchicalCluster:
    if not isinstance(native_cluster, gn.HierarchicalCluster):
        raise TypeError(
            "This class method is only valid for graspologic_native.HierarchicalCluster"
        )
    node_id: Any = identifier.original(native_cluster.node)
    return HierarchicalCluster(
        node=node_id,
        cluster=native_cluster.cluster,
        parent_cluster=native_cluster.parent_cluster,
        level=native_cluster.level,
        is_final_cluster=native_cluster.is_final_cluster,
    )


@beartype
def hierarchical_leiden_new(
    graph: Union[
        List[Tuple[Any, Any, Union[int, float]]],
        nx.Graph,
        np.ndarray,
        scipy.sparse.csr_array,
    ],
    max_cluster_size: int = 1000,
    starting_communities: Optional[Dict[Any, int]] = None,
    extra_forced_iterations: int = 0,
    resolution: Union[int, float] = 1.0,
    randomness: Union[int, float] = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
    is_weighted: Optional[bool] = None,
    weight_default: Union[int, float] = 1.0,
    check_directed: bool = True,
) -> List[HierarchicalCluster]:  # Изменено на ваш собственный класс
    _validate_common_arguments(
        extra_forced_iterations,
        resolution,
        randomness,
        random_seed,
    )
    check_argument(max_cluster_size > 0, "max_cluster_size must be a positive int")

    identifier = _IdentityMapper()
    node_count: int
    edges: List[Tuple[str, str, float]]
    
    if isinstance(graph, nx.Graph):
        components = _bfs_connected_components(graph)
        native_safe_edges = []
        for component in components:
            _, component_edges = _process_component(
                component, graph, identifier, is_weighted, weight_attribute, weight_default
            )
            native_safe_edges.extend(component_edges)
    else:
        raise ValueError("Hierarchical Leiden currently supports only networkx graphs.")

    native_friendly_communities = _community_python_to_native(
        starting_communities, identifier
    )

    hierarchical_clusters_native = gn.hierarchical_leiden(
        edges=native_safe_edges,
        starting_communities=native_friendly_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=extra_forced_iterations + 1,
        use_modularity=use_modularity,
        max_cluster_size=max_cluster_size,
        seed=random_seed,
    )

    result_partitions = []
    for entry in hierarchical_clusters_native:
        partition = HierarchicalCluster(
            node=identifier.original(entry.node),
            cluster=entry.cluster,
            parent_cluster=entry.parent_cluster,
            level=entry.level,
            is_final_cluster=entry.is_final_cluster,
        )
        result_partitions.append(partition)

    return result_partitions
