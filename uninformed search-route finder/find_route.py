# Golay Nie 1001678015
# IMPORTANT: input & heuristic file must have no empty lines after END OF INPUT for program to work!

import sys
from queue import PriorityQueue #fringe, sorted by lowest value

def search(cities, start_city, goal_city, heuristic):
	fringe = PriorityQueue()
	closed = [] #graph search closed set
	nodesexp = nodesgen = maxnodes = 0
	fringe.put([0, start_city])
	while not fringe.empty():
		fringeLen = len(fringe.queue)
		nodesexp += 1
		if maxnodes < fringeLen:
			maxnodes = fringeLen
		node = fringe.get()
		if node[-1] == goal_city:
			return nodesexp, nodesgen, maxnodes, node
		else:
			if node[-1] not in closed:
				closed.append(node[-1])
				for i in cities[node[-1]]:
					nodesgen += 1
					if heuristic == 0:
						fringe.put([node[0]+i[0], node[1:], i[1]]) #uniform cost search
					else:
						fringe.put([node[0]+i[0]+heuristic[i[1]], node[1:], i[1]]) #A* search. only difference is adding heuristic value to cumulative cost
	return nodesexp, nodesgen, maxnodes, None
	
def readInput(input_file):
	data = list() 
	fread = open(input_file, 'r') 
	for line in fread.readlines()[:-1]: #doesn't read last line of file
		data.append(line.split())
	fread.close() 
	return data 

def citylist(data):
	cities = {} #store in a set. no duplicates
	for i in data:
		if i[0] in cities:
			cities[i[0]].append((int(i[2]), i[1]))
		else:
			cities[i[0]] = [(int(i[2]), i[1])]
		if i[1] in cities:
			cities[i[1]].append((int(i[2]), i[0]))
		else:
			cities[i[1]] = [(int(i[2]), i[0])]
	return cities
	
def pathlist(route):
	for i in route:
		if type(i) == list:
			pathlist(i)
		else:
			path.append(i)
	return path	
	
def distance(cities, city1, city2):
	for i in cities[city1]:
		if i[1] == city2:
			return i[0]

def readH(h_file):
	H = {} #store in set, no duplicates
	hread = open(h_file, 'r')
	for line in hread.readlines()[:-1]:
		data = line.split()
		H[data[0]] = int(data[1])
	hread.close()
	return H

if __name__ == '__main__':
	data = readInput(sys.argv[1])
	cities = citylist(data)
	if len(sys.argv) == 4: #uninformed search
		expand, gen, max, route = search(cities, sys.argv[2], sys.argv[3], 0)
	else: #informed search
		expand, gen, max, route = search(cities, sys.argv[2], sys.argv[3], readH(sys.argv[4]))	
	print(f'nodes expanded: {expand}\nnodes generated: {gen}\nmax nodes in memory: {max}')
	if route == None:
		print('distance: infinity\nroute:\nnone')
	else:
		print(f'distance: {route[0]} km\nroute:')
		path = []
		path = pathlist(route[1])
		path.append(route[2])
		for i in range(len(path)-1):
			print(f'{path[i]} to {path[i+1]}, {distance(cities, path[i], path[i+1])} km')