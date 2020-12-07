import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from random import Random
import scipy

# Mapping of start : end spaces of chutes & ladders
# CHUTES_LADDERS = {1: 38, 4: 14, 9: 31, 16: 6, 21: 42, 28: 84, 36: 44,
#                   47: 26, 49: 11, 51: 67, 56: 53, 62: 19, 64: 60,
#                   71: 91, 80: 100, 87: 24, 93: 73, 95: 75, 98: 78}
# rolls = 6
# win_at = 100

CHUTES_LADDERS = {2:6, 8:3}
rolls = 3
win_at = 9

number_of_times = 1000000


def cl_markov_matrix(max_roll=rolls, jump_at_end=False, max=win_at):
    # Create the basic transition matrix:
    mat = np.zeros((max + 1, max + 1))
    for i in range(max + 1):
        for j in range(i + 1, i + max_roll + 1):
            if j > max:
                mat[i][-1] += (1. / max_roll)
                continue
            if j in CHUTES_LADDERS:
                mat[i][CHUTES_LADDERS[j]] += (1. / max_roll)
            else:
                mat[i][j] += (1. / max_roll)
    mat = np.delete(mat, list(CHUTES_LADDERS.keys()), axis=1)
    mat = np.delete(mat, list(CHUTES_LADDERS.keys()), axis=0)
    return mat


x = cl_markov_matrix(max_roll=rolls, max=win_at)
# x = cl_markov_matrix()
s = x[:, -1]
s = s[:-1]
x = np.delete(x, -1, axis=1)
x = np.delete(x, -1, axis=0)
y = np.identity(x.shape[0])
z = np.ones(x.shape[0])
f = np.linalg.inv(y - x).dot(z)
print("Expected number of turns from state 0 computed from fundamental matrix: "+ str(f[0]))


def simulate_cl_game(rseed=None, max_roll=6, max = 100):
    rand = Random(rseed)
    position = 0
    turns = 0
    while position < max:
        turns += 1
        roll = rand.randint(1, max_roll)

        # if the roll takes us past square 100, we don't move
        if position + roll > max:
            return turns

        # otherwise, move the position according to the roll
        position += roll

        # go up/down any chute/ladder
        position = CHUTES_LADDERS.get(position, position)
    return turns


sim_games = [simulate_cl_game(max=win_at, max_roll=rolls) for i in range(number_of_times)]
print(scipy.stats.mode(sim_games))
print(np.mean(sim_games))
print(np.median(sim_games))
plt.hist(sim_games, bins=range(50))
plt.xlabel('Number of Turns to win')
plt.title('Simulated Lengths of Chutes & Ladders Games '+str(number_of_times)+ ' times')
print("Min Count: " + str(sim_games.count(3)))
plt.show()

plt.hist(sim_games, bins=range(50), cumulative=True)
plt.grid(True)
plt.xlabel('Number of Turns to win')
plt.title('CDF Simulated Lengths of Chutes & Ladders Games '+str(number_of_times)+ ' times')
print("Min Count: " + str(sim_games.count(3)))
plt.show()

def simulate_cl_game2(rseed=None, max_roll=6, max = 100):
    rand = Random(rseed)
    position1 = 0
    position2 = 0

    while position1 < max and position2 < max:

        roll1 = rand.randint(1, max_roll)
        roll2 = rand.randint(1, max_roll)

        # if the roll takes us past square 100, we don't move
        if position1 + roll1 > max:
            return 1
        if position2 + roll2 > max:
            return 2

        # otherwise, move the position according to the roll
        position1 += roll1
        position2 += roll2

        # go up/down any chute/ladder
        position1 = CHUTES_LADDERS.get(position1, position1)
        position2 = CHUTES_LADDERS.get(position2, position2)
    if position1 == max:
        return 1
    if position2 == max:
        return 2
    return 0


sim_games = [simulate_cl_game2(max=win_at, max_roll=rolls) for i in range(number_of_times)]
print(scipy.stats.mode(sim_games))
print(np.mean(sim_games))
print(np.median(sim_games))
plt.hist(sim_games, bins=range(15))
plt.xlabel('Number of Turns to win')
plt.title('Player1 vs Player2 '+str(number_of_times)+ ' times')
plt.show()
occ  = sim_games.count(1)
print("Player 1 wins: " + str(sim_games.count(1)))
print("Player 2 wins: " + str(sim_games.count(2)))
print("%chance of Player1 winning: "+str(float(occ/number_of_times)*100))
