# Particle Swarm Optimization Single Cell inference (PSOSC) tool - Cancer progression inference


## The General Idea

The goal is to find an optimal tree given the input matrix.

### Procedure
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

In the first part we use just 2/3 operations, in the second part just 0/1 operations.\
We switch from the first to the second part when:
  - 3/4 of maxtime execution
  - 3/4 of total iterations (if iterations given in input)
  - stuck on a fitness value (if iterations not given in input)

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

**Required input Parameters**
- `--infile [STRING]`: Matrix input file

**Optional input parameters**
- `--mutfile [STRING]`: Path of the mutation names. If not used, then the mutations will be named progressively from 1 to mutations.
- `--particles [INT]`: Number of particles to use for PSO. If not used or zero, it will be number of CPU cores [default: 0]
- `--iterations [INT]`: Number of iterations. If not used or zero, PSO will stop when stuck on a best fitness value (or after maxtime of total execution) [default: 0].
- `--alpha [FLOAT]`: False negative rate [default: 0.15].
- `--beta [FLOAT]`: False positive rate [default: 0.00001].
- `--gamma [FLOAT/STRING]`: Loss rate for each mutation (single float for every mutations or file with different rates) [default: 1].
- `--k [INT]`: K value of Dollo(k) model used as phylogeny tree [default: 3].
- `--maxdel [INT]`: Maximum number of total deletions allowed [default: 5].
- `--tolerance [FLOAT]`: Minimum relative improvement (between 0 and 1) in the last 200 iterations in order to keep going [default: 0.005].
- `--maxtime [INT]`: Maximum time (in seconds) of total PSO execution [default: 300].
- `--multiple [LIST(INT)]`: Multiple runs of the program, with the different number of particles given in input (integers separated by spaces) [default: None].


## Examples

```shell
(env) $ python3 psosc.py --infile "data/hou.txt"
(env) $ python3 psosc.py --infile "data/gawad2.txt" --particles 10 --iterations 20 --k 3 --maxdel 5 --mutfile "data/gawad2_mut.txt"
```


## Info
Author: Leonardo Riva - Università degli studi Milano Bicocca\
Project based on: https://github.com/IAL32/pso-cancer-evolution
