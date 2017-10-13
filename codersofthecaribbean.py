import sys
import math
import numpy as np
import random
from copy import deepcopy
from time import time

### HEXAGRID CODE ###

def cube_to_oddr(cubecoords):
	(x, y, z) = cubecoords
	col = x + (z - (z&1)) // 2
	row = z
	return np.array((col, row))

def oddr_to_cube(hexcoords):
	(col, row) = hexcoords
	x = col - (row - (row&1)) // 2
	z = row
	y = -x-z
	return np.array((x, y, z))

cube_directions = [np.array(x) for x in
	[(1, -1,  0), (1,  0, -1), ( 0, 1, -1),
	(-1, 1,  0), (-1,  0, 1), ( 0, -1, 1)]
]

def get_cube_neighbor(cubecoords, direction):
	return cubecoords + cube_directions[direction]

def cube_distance(a, b):
	return max(abs(a-b)) / 2

### GA CODE ###

class Individual:
	def __init__(self, nr_sequences, rand_funcs):
		self.genome = np.array([[f() for f in rand_funcs] for seq in range(nr_sequences)]).flatten()

class Simulation:
	@staticmethod
	def get_rand_func(gt):
		if gt == "bit":
			return lambda: random.random() < 0.5
		elif gt == "float":
			return random.random
		elif gt[:3] == "int":
			max_nr = int(gt[3:])
			return lambda: random.randint(0,max_nr-1)
		else:
			raise ValueError("Unknown gene type")

	def __init__(self, nr_individuals, mutation_rate, keep_rate, discard_rate, nr_sequences, gene_types, ranking_function, ranking_function_args=[]):
		self.nr_individuals = nr_individuals
		self.mutation_rate = mutation_rate
		self.keep_rate = keep_rate
		self.keep_nr = int(nr_individuals * keep_rate)
		self.discard_rate = discard_rate
		self.discard_nr = int(nr_individuals * discard_rate)
		self.nr_sequences = nr_sequences
		self.population = []
		self.gene_types = gene_types
		self.sequence_length = len(gene_types)
		self.rand_funcs = []
		for gt in self.gene_types:
			self.rand_funcs.append(Simulation.get_rand_func(gt))
		for i in range(nr_individuals):
			self.population.append(Individual(nr_sequences, self.rand_funcs))
		self.ranking_function = ranking_function
		self.ranking_function_args = ranking_function_args

	def iterate(self, generations):
		for gen in range(generations):
			to_crossover = self.population[self.discard_nr:-self.keep_nr]
			random.shuffle(to_crossover)

			newpop = self.population[-self.keep_nr:-1] #not adding the best individual yet, dont want to mutate it

			for i, j in zip(to_crossover[::2],to_crossover[1::2]):
				newpop.extend(self.crossover((i,j)))
			if len(to_crossover) % 2 == 1:
				newpop.append(to_crossover[-1])

			for individual in newpop:
				if random.random() < self.mutation_rate:
					self.mutate(individual)
			newpop.append(self.population[-1])

			for i in range(self.nr_individuals - len(newpop)):
				newindividual = Individual(self.nr_sequences, self.rand_funcs)
				newpop.append(newindividual)

			self.population = self.ranking_function(newpop, self.ranking_function_args)

	def mutate(self, individual):
		seq = int(random.random()*self.nr_sequences)
		seq_gen = int(random.random()*self.sequence_length)
		individual.genome[seq*self.sequence_length+seq_gen] = self.rand_funcs[seq_gen]()

	def crossover(self, pair):
		splitpos = int(random.random()*self.nr_sequences)*self.sequence_length
		i1, i2 = Individual(self.nr_sequences, self.rand_funcs), Individual(self.nr_sequences, self.rand_funcs)
		i1.genome = np.concatenate((pair[0].genome[:splitpos], pair[1].genome[splitpos:]))
		i2.genome = np.concatenate((pair[1].genome[:splitpos], pair[0].genome[splitpos:]))
		return [i1,i2]

### GAME UTILITY ###

class GameState:
	def __init__(self, ships = dict(), barrels = dict(), cannonballs = dict(), mines = dict()):
		self.ships = ships
		self.barrels = barrels
		self.cannonballs = cannonballs
		self.mines = mines

	def getPlayerShips(self):
		return [ship[1] for ship in self.ships.items() if ship[1].owner == PLAYER]

	def getEnemyShips(self):
		return [ship[1] for ship in self.ships.items() if ship[1].owner == ENEMY]

	def get_next_state(self, player_commands, enemy_commands):
		new_gs = deepcopy(self)
		print(new_gs, file=sys.stderr)
		'''
		for ship in new_player_ships:
			ship.rum -= 1
		for ship in new_enemy_ships:
			ship.rum -= 1
		for i, command in enumerate(player_commands):
			if command[0] is "FIRE":
				launch_coords = new_player_ships[i].coords + cube_directions[new_player_ships[i].rotation]
				target = np.array(command[1:])
				#print(launch_coords, target, file=sys.stderr)
				dist = cube_distance(launch_coords, target)
				if dist <= 10:
					turns_to_impact = int(1 + cube_distance(launch_coords, target)/3) #ROUNDING CORRECT?
					new_cannonballs.append(Cannonball(-1, np.array(command[1:3]), i, turns_to_impact))
				#print(new_cannonballs, file=sys.stderr)
		'''
		'''
		One game turn is computed as follows:

		The amount of rum each ship is carrying is decreased by 1 unit.
		The players' commands are applied (spawning of cannon balls, mines and ship movement).
		Ships move forward by the same number of cells as their speed.
		Ships turn.
		Damage from cannon balls is computed.
		Elimination of ships with no more rum.

		If at any point during its movement a ship shares a cell with a barrel of rum, the ship collects
		that rum. In the same way, if a ship touches a mine, that mine explodes and the loss of rum is
		applied immediately.
		'''
		pass #FIXA HÄR NÄSTA. SE UPP MED COPY/DEEPCOPY OSV
		# kvar efter get next state:
		# - ranking som ska vara if-baserad till stor del. smarta funktioner som kollar
		#   chansen att en båt tar stryk t.ex. så att de kan skjuta kanonkulor bredvid
		#   varandra.
		#   en båt med lite rom är viktigare att den inte får fylla på. straffa kvadradiskt
		# - simulera motparten först, sedan själv. motparten antar att våra kommandon är WAIT
		# - antal turns måste tillåta en kanonkula att träffa ganska långt bort

class Ship:
	def __init__(self, ID, coords, rotation, speed, rum, owner, cannon_cooldown = 0, mine_cooldown = 0):
		self.ID = ID
		self.coords = coords
		self.rotation = rotation
		self.speed = speed
		self.rum = rum
		self.owner = owner
		self.cannon_cooldown = cannon_cooldown
		self.mine_cooldown = mine_cooldown

class Barrel:
	def __init__(self, ID, coords, rum):
		self.ID = ID
		self.coords = coords
		self.rum = rum

class Cannonball:
	def __init__(self, ID, target_coords, fired_by, turns_to_impact):
		self.ID = ID
		self.target_coords = coords
		self.fired_by = fired_by
		self.turns_to_impact = turns_to_impact

class Mine:
	def __init__(self, ID, coords):
		self.ID = ID
		self.coords = coords

def ranking(population, args):
	# args are commands for other party
	return population
	#return ranked population, best last

### CONSTANTS AND INITIALIZATION ###

ENEMY = 0
PLAYER = 1
COLS = 21
ROWS = 23
SHIP_LEN = 3
SHIP_WIDTH = 1

genome_rand_funcs = ["int5","int3","int3"]
sequence_length = len(genome_rand_funcs)
nr_boats = None
gene_to_move_type = {
	0: "MOVE",
	1: "FIRE",
	2: "MINE",
	3: "SLOWER",
	4: "WAIT"
}
gene_to_coord_offset = lambda gene: gene-1

player_cannonball_cooldowns = dict()
player_mine_cooldowns = dict()

### GAME LOOP ###

while True:

	### CREATE GAME STATE ###

	GS = GameState(ships=dict(), barrels=dict(), cannonballs=dict(), mines=dict())

	my_ship_count = int(input())  # the number of remaining ships
	if nr_boats is None:
		nr_boats = my_ship_count
	entity_count = int(input())  # the number of entities (e.g. ships, mines or cannonballs)

	for i in range(entity_count):
		entity_id, entity_type, x, y, arg_1, arg_2, arg_3, arg_4 = input().split()
		entity_id = int(entity_id)
		x = int(x)
		y = int(y)
		arg_1 = int(arg_1)
		arg_2 = int(arg_2)
		arg_3 = int(arg_3)
		arg_4 = int(arg_4)
		coords = oddr_to_cube((x,y))

		if entity_type == "SHIP":
			GS.ships[entity_id] = Ship(entity_id, coords, arg_1, arg_2, arg_3, arg_4)
			if arg_4:
				if entity_id not in player_cannonball_cooldowns:
					player_cannonball_cooldowns[entity_id] = 0
				else:
					GS.ships[entity_id].cannon_cooldown = player_cannonball_cooldowns[entity_id]
				if entity_id not in player_mine_cooldowns:
					player_mine_cooldowns[entity_id] = 0
				else:
					GS.ships[entity_id].mine_cooldown = player_mine_cooldowns[entity_id]
		elif entity_type == "BARREL":
			GS.barrels[entity_id] = Barrel(entity_id, coords, arg_1)
		elif entity_type == "CANNONBALL":
			GS.cannonballs[entity_id] = Cannonball(entity_id, coords, arg_1, arg_2)
		elif entity_type == "MINE":
			GS.mines[entity_id] = Mine(entity_id, coords)
		else:
			raise ValueError

	for ship in GS.getPlayerShips():
		ship.cannon_cooldown = player_cannonball_cooldowns[ship.ID]
		ship.mine_cooldown = player_mine_cooldowns[ship.ID]

	### RUN GENETIC ALGORITHM AND GET COMMANDS FOR ENEMY ###

	# improve this by remembering what player planned to do last time
	enemy_sim = Simulation(10, 0.05, 0.2, 0.2, nr_boats, genome_rand_funcs, ranking, ranking_function_args=["WAIT"]*3)
	start = time()
	while time()-start < 0.01:
		enemy_sim.iterate(1)
	enemy_commands = enemy_sim.population[-1].genome[:sequence_length]

	### RUN GENETIC ALGORITHM AND GET COMMANDS FOR PLAYER ###

	### APPLY COMMANDS AND SET COOLDOWNS ###

	for ship in GS.getPlayerShips():
		#if player_cannonball_cooldowns[ship.ID] <= 0:
		#	print("FIRE", *cube_to_oddr(GS.getEnemyShips()[-1].coords), "BOOM!")
		#	player_cannonball_cooldowns[ship.ID] = 2
		#else:
		barreldists = dict()
		for barrel in GS.barrels:
			barreldists[barrel] = cube_distance(GS.barrels[barrel].coords, ship.coords)
		#print(barreldists, file=sys.stderr)
		if len(barreldists) > 0:
			closest = min(barreldists, key=barreldists.get)
			print("MOVE", *cube_to_oddr(GS.barrels[closest].coords), "YARR YE CRABS!")
		else:
			 loc = (random.randint(1,19), random.randint(1,19))
			 print("MOVE", *loc, "YARR YE CRABS!")

	for ship_id in player_cannonball_cooldowns:
		player_cannonball_cooldowns[ship_id] -= 1
	for ship_id in player_mine_cooldowns:
		player_mine_cooldowns[ship_id] -= 1
