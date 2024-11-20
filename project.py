'''
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
'''
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
    """
    edges = set()
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
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


def write_graph_to_file(matrix: list[list[int]], filename: str, is_directed: bool = False):
    """
    Writes a graph from matrix to a text file

    Args:
        matrix: Matrix of the graph
        filename: The output file
        is_directed: True if the graph is directed, false if undirected
    """
    with open(filename, 'w', encoding='utf-8') as file:
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
                        edge = tuple(sorted([i+1, j+1]))
                        if edge not in written_edges:
                            file.write(f"{edge[0]},{edge[1]}\n")
                            written_edges.add(edge)

# if __name__ == "__main__":
    # loaded_matrix = read_graph("graph.csv", is_directed=False)
    # print(loaded_matrix)
    # write_graph_to_file(read_graph("graph.csv", is_directed=False), 'new_file.csv',True)

