# Particle Swarm Optimization Single Cell inference (PSOSC) tool - Cancer progression inference

## The General Idea

The aim is to find an optimal tree given the input matrix.

### Pseudo-Code

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
- `--particles [INT]`: Number of particles to use for PSO. If not used or zero, it will be estimated based on the number of particles and cells [default: 0]
- `--iterations [INT]`: Number of iterations. If not used or zero, PSO will stop when stuck on a best fitness value (or after around 3 minutes of total execution) [default: 0].
- `--alpha [FLOAT]`: False negative rate [default: 0.15].
- `--beta [FLOAT]`: False positive rate [default: 0.00001].
- `--gamma [FLOAT/STRING]`: Loss rate for each mutation (single float for every mutations or file with different rates) [default: 0.5].
- `--k [INT]`: K value of Dollo(k) model used as phylogeny tree [default: 3].
- `--maxdel [INT]`: Maximum number of total deletions allowed [default: 10].
- `--maxtime [INT]`: Maximum time (in seconds) of total PSO execution [default: 300].
- `--multiple [LIST(INT)]`: Multiple runs of the program, with the different number of particles given in input (integers separated by spaces) [default: None].

## Example

```shell
(env) $ python3 psosc.py --infile "data/gawad2.txt" --particles 10 --iterations 20 --w 1 --c1 1 --c2 1 --k 3 --maxdel 5 --mutfile "data/gawad2_mut.txt"
```

## Info
Project based on: https://github.com/IAL32/pso-cancer-evolution
