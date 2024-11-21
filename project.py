"""
Для файлу з вмістом:
        1,2
        2,3
        3,4
        1,4
    функція read_graph_from_file виведе наступне:

Для орієнтованого графу:
[
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
]

Для неорієнтованого графу:
[
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
]
"""


def read_graph(filename: str, is_directed: bool = False) -> list[list[int]]:
    """
    Reads a graph from a text file and returns it as a matrix

    Args:
        filename: Path to the file
        is_directed: True if the graph is directed, false if undirected
    Returns:
        Matrix where 1 indicates connected vertices,
                         0 indicates unconnected vertices

    Example file format:
        1,2
        2,3
        3,4
        1,4
        Each line represents an edge with two vertices separated by a comma

    >>> loaded_matrix = read_graph("graph.csv", is_directed=False)
    >>> print(loaded_matrix)
    [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
    """
    edges = set()
    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: {line}")

                try:
                    v1, v2 = map(int, parts)
                except ValueError:
                    raise ValueError(f"Vertices must be integers: {line}")

                if v1 <= 0 or v2 <= 0:
                    raise ValueError(f"Vertex numbers must be positive: {line}")

                edges.add((v1, v2))
                max_vertex = max(v1, v2)

    except FileNotFoundError:
        print(f"File {filename} not found")
        return []

    matrix = [[0 for _ in range(max_vertex)] for _ in range(max_vertex)]

    for v1, v2 in edges:
        v1 -= 1
        v2 -= 1
        matrix[v1][v2] = 1
        if not is_directed:
            matrix[v2][v1] = 1

    return matrix


def write_graph_to_file(
    matrix: list[list[int]], filename: str, is_directed: bool = False
):
    """
    Writes a graph from matrix to a text file

    Args:
        matrix: Matrix of the graph
        filename: The output file
        is_directed: True if the graph is directed, false if undirected

    >>> graph = [[0, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]]
    >>> write_graph_to_file(graph, "new_graph.csv")
    >>> with open("new_graph.csv", "r", encoding="utf-8") as file:
    ...     print(file.read())
    1,2\n\
    1,3\n\
    1,4\n\
    2,3\n\
    3,3\n\
    3,4
    """
    with open(filename, "w", encoding="utf-8") as file:
        n = len(matrix)
        if is_directed:
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] == 1:
                        file.write(f"{i+1},{j+1}\n")
        else:
            written_edges = set()
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] == 1:
                        edge = tuple(sorted([i + 1, j + 1]))
                        if edge not in written_edges:
                            file.write(f"{edge[0]},{edge[1]}\n")
                            written_edges.add(edge)


# if __name__ == "__main__":
# loaded_matrix = read_graph("graph.csv", is_directed=False)
# print(loaded_matrix)
# write_graph_to_file(read_graph("graph.csv", is_directed=False), 'new_file.csv',True)


def find_connectivity(graph: list[list[int]]) -> list[list[int]]:
    """
    Finds all connected components in an undirected graph.

    Args:
        matrix (list[list[int]]): Adjacency matrix of the graph.

    Returns:
        list[list[int]]: A list of connected components, each represented by a list of vertices.

    >>> matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> find_connectivity(matrix)
    [[0, 1, 2]]
    >>> matrix = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    >>> find_connectivity(matrix)
    [[0, 1], [2, 3]]
    """
    pass


def find_strong_connectivity(graph: list[list[int]]) -> list[list[int]]:
    pass


def find_connection_points(graph: list[list[int]]) -> list:
    pass


def find_bridges(graph: list[list[int]]) -> list:
    """
    Finds all bridges in an undirected graph represented by an adjacency matrix.
    A bridge is an edge that, when removed, increases the number of connected components \
    in the graph.

    Args:
        graph (list[list[int]]): Adjacency matrix representing the graph.

    Returns:
        list[tuple[int, int]]: A list of bridges, where each bridge is represented as a \
        tuple of two integers (the endpoints of the edge).

    Examples:
    >>> matrix = [[0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], \
[0, 0, 0, 1, 0]]
    >>> find_bridges(matrix)
    [(1, 2), (3, 4)]
    >>> matrix = [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], \
[1, 0, 0, 1, 0]]
    >>> find_bridges(matrix)
    []
    >>> matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0],]
    >>> find_bridges(matrix)
    [(0, 1), (1, 2)]
    """
    bridges = []
    initial_components = find_connectivity(graph)

    n = len(graph)
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] == 1:
                graph[i][j], graph[j][i] = 0, 0
                new_components = find_connectivity(graph)
                if len(new_components) > len(initial_components):
                    bridges.append((i, j))
                graph[i][j], graph[j][i] = 1, 1

    return bridges


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())