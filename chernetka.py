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
    >>> find_connection_points(matrix) == {0, 1, 5}
    True
    """
    starting_node = 0
    list_of_deep_search = [0 for _ in range(len(graph))]
    list_of_lows = [0 for _ in range(len(graph))]
    depth = 1
    current_node = starting_node
    check_if_lows_are_full = False
    stack = []
    visited = []
    while not check_if_lows_are_full:
        print(f'{current_node=}')
        print(f'{depth=}')
        print(f'{stack=}')
        print(f'{list_of_deep_search=}')
        print(f'{list_of_lows=}')
        if depth == 10:
            break
        if list_of_deep_search[current_node] == 0:
            list_of_deep_search[current_node] = depth
            depth += 1
            # stack.append(current_node)
            visited.append(current_node)
        else:
            for index, j in enumerate(graph[current_node]):
                if j and index not in visited:
                    print(f'j ={j}')
                    stack.append(current_node)
                    current_node = index
                    break
            else:
                temp_list_of_adjacent_points = []
                if not list_of_lows[current_node]:
                    for ind, k in enumerate(graph[current_node]):
                        if k:
                            temp_list_of_adjacent_points.append(list_of_deep_search[ind])
                    list_of_lows[current_node] = min(temp_list_of_adjacent_points)
                current_node = stack.pop()

        if 0 not in list_of_lows:
            check_if_lows_are_full = True
    print(list_of_deep_search)
    print(list_of_lows)

matrix = [
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0]
]

find_connection_points(matrix)