# Particle Swarm Optimization for Cancer Evolution 2.0

## Pseudo-Code

```
for each particle
    initialize position particle
    initialize velocity particle
end for

// iteration
k = 1
do
    for each particle
        calculate fitness value
        if < the fitness value is better than the best fitness value (pbest) in history >
            set current value as the new pbest
        end if
    end for

    choose the particle with the best fitness value of all the particles as the gbest

    for each particle
        calculate particle velocity according to:
            v_i(k + 1) = w * v_i(k) + c1 * rand(p_i - x_i) + c2 * rand(p_g - x_i)
        calculate particle position according to:
            x_i(k + 1) = x_i(k) + v_i(k + 1)
    end for
    k = k + 1
while maximum iterations or minimum error criteria is not attained
```

## The General Idea

The aim is to find an optimal tree given the input matrix.

## Setup

I recommend creating a `virtualenv` with `python3`, because that's what I used:
```shell
$ virtualenv env -p `which python3`
$ source ./env/bin/activate
```


Then install every package necessary:

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

**Input Parameters (required)**
- `--infile [STRING]`: Matrix input file

**Input Parameters (optional)**
- `--mutfile [STRING]`: Path of the mutation names. If this parameter is not used, then the mutations will be named progressively from 1 to mutations.
- `--particles [INT]`: Number of particles to use for PSO [default: 5]
- `--iterations [INT]`: Number of iterations [default: 3].
- `--alpha [FLOAT]`: False negative rate [default: 0.15].
- `--beta [FLOAT]`: False positive rate [default: 0.00001].
- `--gamma [FLOAT/STRING]`: Loss probability for each mutations [default: 1].
- `--c1 [FLOAT]`: Learning factor for particle best [default: 0.25].
- `--c2 [FLOAT]`: Learning factor for swarm best [default: 0.75].
- `--k [INT]`: K value of Dollo(k) model used as phylogeny tree [default: 3].
- `--maxdel [INT]`: Maximum number of total deletions allowed [default: 10].

## Example

Example:

```shell
(env) $ python3 main.py --infile "data/gawad2.txt" --particles 10 --iterations 20 --c1 1 --c2 1 --maxdel 5 --k 3 --mutfile "data/gawad2_mut.txt"
```
