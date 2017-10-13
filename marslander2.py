'''
Increase search depth:
    - Simplify turn calc if possible. Can i make multiple turns calced at once? Acceleration applied earlier has bigger effect on velocity now, etc.
    - Fitness needs to be simple as shit. And logic. Have we landed? Are we dead? Have we ended cruising? Are we falling fast? 0/1 questions.
      Divide population into subpops depending on these questions. Skip sorting?
      Or: Kill individuals not making it to these steps. Add new dynamically.
    - Save genome from best individual at each step. Prime next iteration. Also: When we find a solution that takes us to the next step,
      accept it as a solution and start evolving the next part instead.
'''

import sys
import math
import random
import numpy as np
from copy import deepcopy
from time import time

#########################################################
# Convenience

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

    def __init__(self, nr_individuals, mutation_rate, keep_rate, discard_rate, nr_sequences, gene_types, ranking_function):
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

            self.population = self.ranking_function(newpop)

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

class Vector(object):
    def __init__(self, values):
        self.__values = tuple(map(float,values))

    def __repr__(self):
        return "<V%i: <" % len(self.__values) + ("%.2f, "*len(self.__values) % self.__values)[:-2] + ">>"

    def __add__(self, v):
        if len(self.__values) != len(v[:]):
            raise ValueError("Dimension mismatch")
        return Vector([a+b for (a,b) in zip(self.__values, v[:])])

    def __sub__(self, v):
        if len(self.__values) != len(v[:]):
            raise ValueError("Dimension mismatch")
        return Vector([a-b for (a,b) in zip(self.__values, v[:])])

    def __abs__(self):
        return math.sqrt(sum([v**2 for v in self.__values]))

    def __div__(self, n):
        return Vector([i/float(n) for i in self.__values])

    def __mul__(self, n):
        return Vector([i*n for i in self.__values])

    def __rmul__(self, n):
        return Vector([i*n for i in self.__values])

    def __getitem__(self, i):
        return self.__values[i]

    def __neg__(self):
        return Vector([-i for i in self.__values])

    def dot(self, v):
        if len(self.__values) != len(v[:]):
            raise ValueError("Dimension mismatch")
        return sum([a*b for (a,b) in zip(self.__values, v[:])])

    def scalar_projection(self, v):
        if len(self.__values) != len(v[:]):
            raise ValueError("Dimension mismatch")
        return self.dot(v.norm())

    def vector_projection(self, v):
        if len(self.__values) != len(v[:]):
            raise ValueError("Dimension mismatch")
        return (v.dot(self) / abs(v)**2) * v

    def cross3D(self, v):
        if len(self.__values) != len(v[:]) != 3:
            raise ValueError("Dimension mismatch")
        return Vector    ([ self[1]*v[2]-self[2]*v[1], \
                            self[2]*v[0]-self[0]*v[2], \
                            self[0]*v[1]-self[1]*v[0] ])

    def cross2D(self, v):
        if len(self.__values) != len(v[:]) != 2:
            raise ValueError("Dimension mismatch")
        return self[0]*v[1]-self[1]*v[0]

    def angle2D(self, v):
        if len(self.__values) != len(v[:]) != 2:
            raise ValueError("Dimension mismatch")
        return math.atan2(v[1], v[0]) - math.atan2(self[1], self[0])

    def rotate2D(self, theta):
        if len(self.__values) != 2:
            raise ValueError("Dimension mismatch")
        newx = self[0]*math.cos(theta)-self[1]*math.sin(theta)
        newy = self[0]*math.sin(theta)+self[1]*math.cos(theta)
        return Vector((newx,newy))

    def norm(self):
        length = abs(self)
        if length == 0:
            return self
        return Vector([v/length for v in self.__values])

    def __eq__(self, v):
        if isinstance(v, self.__class__):
            return self.__dict__ == v.__dict__
        return NotImplemented

    def __ne__(self, v):
        if isinstance(v, self.__class__):
            return not self.__eq__(v)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

class Bezier:
    def __init__(self, points):
        self.points = [Vector(p) for p in points]

    def get_value(self, t):
        if t < 0 or t > 1:
            raise ValueError
        point_reduction = self.points
        while len(point_reduction) > 1:
            pr = []
            for i, p in enumerate(point_reduction[:-1]):
                pr.append(p*(1-t)+point_reduction[i+1]*t)
            point_reduction = pr
        return Vector((pr[0][0], pr[0][1]))

class TargetGiver:
    padding = 100

    def __init__(self, points, resolution):
        self.bezier = Bezier(points)
        self.targets = [self.bezier.get_value(t) for t in np.linspace(0, 1, resolution)]
        self.left_to_right = points[0][0] < points[-1][0]

    def get_target(self, x):
        if self.left_to_right:
            for t in self.targets:
                if t[0]-TargetGiver.padding > x:
                    return t
        else:
            for t in self.targets:
                if t[0]+TargetGiver.padding < x:
                    return t

#########################################################
# Constants

G = 3.711
upvector = Vector((0,1))
rightvector = Vector((1,0))
G_vector = Vector((0,-1))*G

simulated_turns = 6
gene_types = ["int31","int3"]
sequence_length = len(gene_types)
genome_len = simulated_turns*sequence_length
target_resolution = 5
angle_to_vector = dict(zip(range(-90,91), [upvector.rotate2D(a*math.pi/180.0) for a in range(-90,91)]))

#########################################################
# Game specific functions

def getLandingInfo():
    for i, point in enumerate(surface[:-1]):
        if surface[i+1][1] == point[1]:
            return [Vector(surface[i]), Vector(surface[i+1])], [point[0],surface[i+1][0]]

def getHeight(x):
    i = 0
    for i in range(surface_n):
        if surface[i+1][0] >= x:
            break
    x1, y1, x2, y2 = surface[i][0], surface[i][1], surface[i+1][0], surface[i+1][1]
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

def above_landing_area(position):
    return landing_area[0] <= position[0] <= landing_area[1]

def next_state(current_state, commands):
    new_angle = min(max(current_state["angle"] + commands["angle"], -90), 90)
    #print(current_state["angle"], commands["angle"], new_angle, file=sys.stderr)
    power = min(max(current_state["power"] + commands["power"], 0), 4)
    thrustvector = angle_to_vector[new_angle]*power
    acceleration = thrustvector+G_vector
    new_velocity = current_state["velocity"] + acceleration
    new_position = current_state["position"] + new_velocity
    new_fuel = current_state["fuel"] - power
    new_landed, new_crashed = False, False
    if new_position[1] <= getHeight(new_position[0]):
        #print("Landing!", new_angle, new_angle==0, new_velocity[1], new_velocity[1] >= -40, abs(new_velocity[0]), abs(new_velocity[0]) <= 20, file=sys.stderr)
        if new_angle == 0 and new_velocity[1] >= -40 and \
            abs(new_velocity[0]) <= 20 and \
            landing_area[0] < new_position[0] < landing_area[1]:
            new_landed = True
            #print("Landed!", file=sys.stderr)
        else:
            new_crashed = True
            #print("Crashed!", file=sys.stderr)
    return {"angle":new_angle,"power":power,"velocity":new_velocity,"position":new_position,"fuel":new_fuel, \
            "landed":new_landed,"crashed":new_crashed}

def gene_to_rel_angle(gene):
    return gene-15

def gene_to_rel_thrust(gene):
    return gene-1

def appr_turns_to_ground(position, speed):
    dist = position[1] - getHeight(position[0])
    return dist / -speed[1]

def get_max_point_before_goal(position):
    h = 0
    x = 0
    for point in surface:
        if (landing_point[0] < point[0] < position[0]) or \
            (landing_point[0] > point[0] > position[0]):
            if point[1] > h:
                h = point[1]
                x = point[0]
    return Vector((x,h))

def ranking(population):
    return sorted(population, key=lambda individual: score_end_game_state(get_end_game_state(individual)))

def get_target(position):
    t = target_giver.get_target(position[0])
    if not t:
        t = landing_point
    return t

def score_end_game_state(game_state):
    t = get_target(game_state['position'])
    wanted_v = (t-game_state['position']).norm()*[20,50][abs(game_state['position'][0]-landing_point[0])>300]
    vdiff_vector = wanted_v - game_state["velocity"]
    velocitydiff = abs(vdiff_vector)
    return [-abs(game_state['angle'])-velocitydiff, 1000+game_state["fuel"]][game_state["landed"]]
          #+ (-abs(game_state['angle']) if not game_state["landed"] else 1000)

def get_end_game_state(individual):
    this_game_state = deepcopy(game_state)
    stop = False
    for i in range(simulated_turns):
        if not stop:
            commands = individual.genome[i*sequence_length:(i+1)*sequence_length]
            if sequence_length == 3:
                nr_turns = int(commands[2]*3)+1 #1 to 3 turns per command
            else:
                nr_turns = 1
            commands = {"angle":gene_to_rel_angle(commands[0]),"power":gene_to_rel_thrust(commands[1])}
            for turns in range(nr_turns):
                if not this_game_state["crashed"] and not this_game_state["landed"]:
                    this_game_state = next_state(this_game_state, commands)
                else:
                    stop = True
                    break
        else:
            break
    return this_game_state

#########################################################
# Initialization

surface = []
surface_n = int(input())
for i in range(surface_n):
    land_x, land_y = [int(j) for j in input().split()]
    surface.append(Vector([land_x, land_y]))

landing_vertices, landing_area = getLandingInfo()
landing_point = Vector([sum(landing_area)/2, getHeight(landing_area[0])])

last_best = None
target_giver = None

#########################################################
# Main loop

while True:
    x, y, h_speed, v_speed, fuel, angle, power = [int(i) for i in input().split()]
    position = Vector((x,y))
    velocity = Vector((h_speed, v_speed))

    if not target_giver:
        first = position
        second = Vector((landing_point[0], y))
        third = landing_point
        points = [first, second, third]
        target_giver = TargetGiver(points, target_resolution)

    game_state = {"angle":angle,"power":power,"velocity":velocity,"position":position,"fuel":fuel, \
            "landed":False,"crashed":False}

    mySim = Simulation(10, 0.02, 0.2, 0.1, simulated_turns, gene_types, ranking)
    if last_best and simulated_turns > 1:
        for individual in mySim.population:
            individual.genome[:-sequence_length] = last_best.genome[sequence_length:]

    s = time()
    gens = 0
    while (time()-s) < 0.099:
        mySim.iterate(1)
        gens += 1
    best = mySim.population[-1]

    print("Generations: " + str(gens), file=sys.stderr)
    '''
    t = target_giver.get_target(game_state["position"][0])
    print("T: " + str(t), file=sys.stderr)
    wanted_v = (t-game_state['position']).norm()*35
    print("W V: " + str(wanted_v), file=sys.stderr)
    needed_angle = - int(wanted_v.angle2D(upvector) * 180 / math.pi)
    print("needed a: " + str(needed_angle), file=sys.stderr)
    anglediff = abs(needed_angle - game_state["angle"])
    print("anglediff: " + str(anglediff), file=sys.stderr)
    '''

    new_angle = min(max(game_state["angle"] + gene_to_rel_angle(best.genome[0]), -90), 90)
    power = min(max(game_state["power"] + gene_to_rel_thrust(best.genome[1]), 0), 4)
    print(*[str(new_angle), str(power)])

    last_best = best
