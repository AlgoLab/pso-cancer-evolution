# Particle Swarm Optimization Single Cell inference (PSOSC) tool - Cancer progression inference


## The General Idea
The goal is to find an optimal tree given the input matrix.

#### Procedure
  - Create particles
  - Every particle run independently from the others:
      - Inertia movement (random operation)
      - Movement to best particle tree (clade attachment from the best particle tree to the current tree)
      - Movement to best swarm tree (clade attachment from the best swarm tree to the current tree)
      - Update possible new particle best and swarm best
      - Stop when one of the stopping criteria is met

The possible operations are:
  - add new random backmutation (0)
  - remove random backmutation (1)
  - swap two random nodes (2)
  - prune-regraft two random nodes (3)

In the first part we use just 2/3 operations, in the second part 0/1 too.\
We switch from the first to the second part when:
  - 3/4 of maxtime execution
  - 3/4 of total iterations (if iterations given in input)
  - stuck on a fitness value (if iterations not given in input)

For the entire execution, when stuck in a local optimum, some random operation will be performed on every tree, in order to exit it.

The particles stop when:
  - they reach maxtime
  - they reach total iterations (if iterations given in input)
  - they're stuck on a fitness value (if iterations not given in input)


## Setup
I recommend creating a `virtualenv` with `python3` (just for the 1st time):
```shell
$ virtualenv env -p `which python3`
```

Then activate the virtual environment:
```shell
$ source ./env/bin/activate
```

Finally, install every necessary package:
```shell
(env) $ pip install numpy
(env) $ pip install graphviz
(env) $ pip install ete3
(env) $ pip install docopt
(env) $ pip install matplotlib
(env) $ pip install networkx
```

Done!


## Usage
**Required input parameters**
- `-i [STRING]`: Matrix input file
- `-p [INT]`: Number of particles to use for PSO (single or multiple values, separated by commas, for a multiple run).
- `-c [INT]`: Number of CPU cores used for the execution.
- `-k [INT]`: K value of Dollo(k) model used as phylogeny tree.
- `-a [FLOAT/STRING]`: False negative rate in input file or path of the file containing different FN rates for each mutations.
- `-b [FLOAT]`: False positive rate.

**Optional input parameters**
- `-g [FLOAT/STRING]`: Loss rate in input file or path of the file containing different GAMMA rates for each mutations [default: 1].
- `-t [INT]`: Number of iterations (-m argument will be ignored; not used by default).
- `-d [INT]`: Maximum number of total deletions allowed [default: +inf].
- `-e [STRING]`: Path of the mutation names. If not used, mutations will be named progressively from 1 to mutations.
- `-T [FLOAT]`: Tolerance, minimum relative improvement (between 0 and 1) in the last iterations in order to keep going, if iterations are not used [default: 0.005].
- `-m [INT]`: Maximum time (in seconds) of total PSO execution (not used by default).
- `-I [STRING]`: Actual correct matrix, for testing (not used by default).

**Optional execution options**
- `--quiet`: Doesn't print anything (not used by default).
- `--output [STRING]`: Limit the output (files created) to: (image | plot | text_file | all) [default: all]


## Examples
```shell
(env) $ python3 psosc.py -i "data/gawad2.txt" -p 4 -c 2 -k 3 -a 0.15 -b 0.00001 -d 5 -e "data/gawad2_mut.txt"
(env) $ python3 psosc.py -i "./data/simulated/exp1/sim_1_scs.txt" -p 10,20,50 -c 4 -k 3 -a 0.25 -b 0.00001 -I "./data/simulated/exp1/sim_1_truescs.txt" -m 600
```


## Info
Author: Leonardo Riva - Università degli studi Milano Bicocca\
Project based on: https://github.com/IAL32/pso-cancer-evolution
