# DSMC GPU Implementation Project
Project for Parallel and Distributed Scientific Computing DIS at Mississippi 
State University. 


## Original description given with starter code
Example Direct Simulation Monte Carlo (DSMC) program.  Use this
program as a guide to your parallel implementation.  You may use this
code if you wish, but it is also desirable if you find other more
efficient implementations.  The code outputs particles and average cell data
files in the files particles.dat and cells.dat.  A helper program, pextract,
is provided that can extract the particle data to allow it to be plotted using
a 2-D plotting package.

The program can be run in either 2D or 3D modes.  By default it solves
in the 3-d mode with 32x32x32 grid and about 10 particles per cell at
the inflow.  The goal is to solve on at least a 128x128x128 grid using
a parallel implementation.



## Current issue
For some reason, it seems only cells with idx%32 = 0 are being sampled to have
particles in them. This may be because either indexing is broken, moving particles
is broken, the correct number of blocks are not launched (which I find unlikely?), 
or the new particles are being initialized over the old ones. 
Firstly, I should get rid of the side-effects of the debugging print statements...
Maybe only have it sort if it has not been sorted?
Or have a "state" variable that indicates what type of sorting the particles array
is in? Like -1 for not initialized, 0 with new particles in the end chunk, 1 for 
sorted by validity (all valid particles near beginning), 2 for sorted by index, and maybe others later?
No idea how this would be best implemented. 
