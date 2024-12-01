'''A Python library for analyzing graph structures'''
import copy

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
    3,4\n\
    <BLANKLINE>
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


def find_connectivity(graph: list[list[int]]) -> list[list[int]]:  # Sofiia Sychak
    """
    Finds all connected components in an undirected and a directed(weak connectivity) graph.

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

    def is_directed_graph(graph: list[list[int]]) -> bool:
        """
        Checks if the graph is directed.

        Args:
            matrix (list[list[int]]): Adjacency matrix of the graph.
        Returns:
            True or False: bool dtatement that represent whether the graph is directed.
        >>> matrix = [
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
        ]
        >>> is_directed_graph(matrix)
        True
        """
        for i, row in enumerate(graph):
            for j, value in enumerate(row):
                if value != graph[j][i]:
                    return True
        return False

    def make_undirected(graph):
        """
        Converts a directed graph into an undirected one for weak connectivity.

        Args:
            matrix (list[list[int]]): Adjacency matrix of the graph.
        Returns:
            matrix (list[list[int]]): Renewed the graph that is undirectted now.
        >>> matrix = [
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
        ]
        >>> make_undirected(matrix)
            [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
        ]
        """
        for i, row in enumerate(graph):
            for j, value in enumerate(row):
                if value == 1 or graph[j][i] == 1:
                    graph[i][j] = graph[j][i] = 1
        return graph

    visited = [False] * len(graph)
    components = []

    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor, connected in enumerate(graph[node]):
            if connected == 1 and not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(len(graph)):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)

    if is_directed_graph(graph):
        graph = make_undirected(graph)

    return components


def transp_graph(graph):
    """
    Створює транспонований граф (змінює напрямки всіх ребер)
    :param graph: граф
    :return: транспонований граф
    >>> transpose_graph([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], \
[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
    [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]]
    """
    vertices = len(graph)
    transp = [[0] * vertices for _ in range(vertices)]

    for num1 in range(vertices):
        for num2 in range(vertices):
            if graph[num1][num2]:
                transp[num2][num1] = 1
    return transp


def dfs_kosaraju(graph, vert, visited, stack):
    """
    Перший прохід DFS для алгоритму Косараджу
    Заповнює стек у порядку завершення обходу
    :param graph: граф у вигляді матриці
    :param vert: вершина
    :param visited: список вершин, які були пройдені
    :param stack: стек, у який записуються вершини
    """
    visited[vert] = True

    for vert1 in range(len(graph)):
        if graph[vert][vert1] == 1 and visited[vert1] is False:
            dfs_kosaraju(graph, vert1, visited, stack)

    stack.append(vert)


def find_strong_connectivity_kosaraju(graph):
    """
    Знаходить компоненти сильної зв'язності у графі
    Повертає список компонент, де кожна компонента - це список вершин
    :param graph: граф у вигялді матриці
    :return: список компонент сильної зв'язності
    >>> find_strong_connectivity([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], \
[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
    [[0, 1, 2], [3, 4]]
    """
    vertices = len(graph)
    visited = [False] * vertices
    stack = []

    for vert in range(vertices):
        if visited[vert] is False:
            dfs_kosaraju(graph, vert, visited, stack)

    transp = transp_graph(graph)
    visited = [False] * vertices
    comps = []
    while stack:
        vert = stack.pop()
        if visited[vert] is False:
            curr_comp = []
            dfs_kosaraju(transp, vert, visited, curr_comp)
            comps.append(sorted(curr_comp))
    return sorted(comps)



def deep_first_search(graph: list[list[int]], strating_point: int = 0) -> list:
    """
    Function that implements depth-first search (DFS).

    :param graph: list[list[int]], adjacency matrix of the graph
    :param starting_point: int, The point to start the search with
    :return: list of visited nodes

    >>> matrix = [
    ...     [0, 1, 1, 1, 0],
    ...     [1, 0, 1, 0, 0],
    ...     [1, 1, 0, 0, 1],
    ...     [1, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0]
    ... ]
    >>> deep_first_search(matrix, 0)
    [0, 1, 2, 4, 3]
    """
    visited = []
    stack = [strating_point]

    while stack:
        temp_stack = []
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            for neighbor, is_connected in enumerate(graph[node]):
                if is_connected and neighbor not in visited:
                    temp_stack.append(neighbor)
            stack.extend(list(reversed(temp_stack)))
    return visited


def find_connection_points(graph: list[list[int]]) -> set:
    """
    Function that searches for articulation points in graph

    :param graph: list[list[int]], Adjacency matrix of graph
    :return: list, List of all articulation points in graph

    >>> matrix = [
    ...     [0, 1, 1, 1, 0],
    ...     [1, 0, 1, 0, 0],
    ...     [1, 1, 0, 0, 1],
    ...     [1, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0]
    ... ]
    >>> find_connection_points(matrix) == {0, 2}
    True
    >>> matrix = [
    ...     [0, 1, 1, 1, 0, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0, 0, 0],
    ...     [1, 1, 0, 0, 0, 0, 0, 0],
    ...     [1, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 1, 0, 0, 0, 1, 0, 1],
    ...     [0, 0, 0, 0, 1, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 1, 0, 1],
    ...     [0, 0, 0, 0, 1, 0, 1, 0]
    ... ]
    >>> find_connection_points(matrix) == {0, 1, 4}
    True
    """
    articulation_points = set()
    n = len(graph)
    original_components_of_conectivity = len(find_connectivity(graph))
    for i in range(n):
        graph_copy = copy.deepcopy(graph)
        for j in range(n):
            graph_copy[j].pop(i)
        graph_copy.pop(i)
        new_components_of_conectivity = len(find_connectivity(graph_copy))
        if new_components_of_conectivity > original_components_of_conectivity:
            articulation_points.add(i)
    return articulation_points

def find_connection_points_optimized(graph: list[list[int]]) -> set:
    """
    Function that searches for articulation points in graph

    :param graph: list[list[int]], Adjacency matrix of graph
    :return: list, List of all articulation points in graph

    >>> matrix = [
    ...     [0, 1, 1, 1, 0],
    ...     [1, 0, 1, 0, 0],
    ...     [1, 1, 0, 0, 1],
    ...     [1, 0, 0, 0, 0],
    ...     [0, 0, 1, 0, 0]
    ... ]
    >>> find_connection_points_optimized(matrix) == {0, 2}
    True
    >>> matrix = [
    ...     [0, 1, 1, 1, 0, 0, 0, 0],
    ...     [1, 0, 1, 0, 1, 0, 0, 0],
    ...     [1, 1, 0, 0, 0, 0, 0, 0],
    ...     [1, 0, 0, 0, 0, 0, 0, 0],
    ...     [0, 1, 0, 0, 0, 1, 0, 1],
    ...     [0, 0, 0, 0, 1, 0, 1, 0],
    ...     [0, 0, 0, 0, 0, 1, 0, 1],
    ...     [0, 0, 0, 0, 1, 0, 1, 0]
    ... ]
    >>> find_connection_points_optimized(matrix) == {0, 1, 4}
    True
    """
    starting_node = 0
    articulation_points = set()
    n = len(graph)
    current_node = starting_node
    stack = []
    number = 1
    order_of_nodes = [0]*n
    lows = [n]*n
    stack.append(current_node)
    order_of_nodes[current_node] = number
    number += 1
    while stack:
        neighbor = -1
        for i in range(n):
            if i == current_node:
                continue
            if graph[current_node][i] == 0:
                continue
            if order_of_nodes[i] != 0:
                if lows[current_node] > order_of_nodes[i]:
                    lows[current_node] = order_of_nodes[i]
                continue
            else:
                neighbor = i
                break
        if neighbor == -1:
            current_low = lows[current_node]
            if len(stack) == 1:
                current_node = stack.pop()
                break
            else:
                stack.pop()
                current_node = stack[-1]
            if lows[current_node] > current_low:
                lows[current_node] = current_low
            continue
        current_node = neighbor
        order_of_nodes[current_node] = number
        number += 1
        stack.append(current_node)
    number_of_children_of_starting_node = 0
    for j in graph[starting_node]:
        if j != 0:
            number_of_children_of_starting_node += 1
    if number_of_children_of_starting_node > 1:
        articulation_points.add(starting_node)
    for i in range(n):
        for j in range(n):
            if graph[i][j] and lows[j] == order_of_nodes[i]:
                articulation_points.add(i)
                break
    return articulation_points


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
    >>> matrix1 = [[0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], \
[0, 0, 0, 1, 0]]
    >>> find_bridges(matrix1)
    [(1, 2), (2, 3), (2, 4), (4, 5)]
    >>> matrix2 = [[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], \
[0, 0, 0, 1, 0]]
    >>> find_bridges(matrix2)
    [(3, 4), (4, 5)]
    >>> matrix3 = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], \
[0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> find_bridges(matrix3)
    [(3, 4)]
    >>> matrix4 = [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], \
[1, 0, 0, 1, 0]]
    >>> find_bridges(matrix4)
    []
    >>> matrix5 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> find_bridges(matrix5)
    [(1, 2), (2, 3)]
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
                    bridges.append((i + 1, j + 1))
                graph[i][j], graph[j][i] = 1, 1

    return bridges


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
