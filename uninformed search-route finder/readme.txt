Finds shortest route. See image for map.

Program uses Python 3.7

Structure: Uses a search function with graph search (closed set of visited nodes)
that implements either a uniform cost search or A* search formula depending on input.
A PriorityQueue automatically sorts the Fringe by lowest value.
Other functions to read input files and organize cities and paths.

run "python find_route.py input_filename.txt origin_city destination_city heuristic_filename.txt"

city names are case sensitive
