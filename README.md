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

## Example

Example:

```shell
(env) $ python3 main.py --infile "data/gawad2.txt" --particles 10 --iterations 20 --c1 1 --c2 1 --d 10 --k 10 --mutfile "data/gawad2_mut.txt"
```
