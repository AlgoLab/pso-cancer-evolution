




cores = 2
multiple_runs = None
particles =1



b1 = (multiple_runs and any(cores > particles for particles in multiple_runs))
b2 = (not(multiple_runs) and cores > particles)

print("b1: " + str(b1))
print("b2: " + str(b2))

if (b1 or b2):
    print("Error! Cores cannot be more than particles")
else:
    print("ok")



if (multiple_runs and any(cores > particles for particles in multiple_runs)) or (not(multiple_runs) and cores > particles):
    print("ERROREEEEEEE")
else:
    print("OOOOOK")
