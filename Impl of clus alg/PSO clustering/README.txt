Requirments to compile and run:
OpenMPI or Mpich
GCC
--------------------------------------------
Commands to compile:

mpicc MPI_PSOClust.c -o3 -fopenmp -o program.exe
--------------------------------------------
Parameters are located on the beggining of source code, example of parameters to cluster IRIS:
//data specific parameters
#define KMEANS 3 //number of cluster groups
#define DIMENSIONS 4 //number of dimensions of data
#define DATA_ROWS 150

//runs specific parameters
#define SWARM_SIZE 200
#define MAX_ITERATIONS 10000
#define THREAD_COUNT 6 //count of thread used by OpenMP

// one of these three parameters must be true, in this example rooted euclidean distance is used as distance function.
#define USE_ROOTED_EUCLIDEAN_DISTANCE 1
#define USE_EUCLIDEAN_DISTANCE 0
#define USE_MANHATTAN_DISTANCE 0
--------------------------------------------
To cluster data, run compiled program with parameters. Use program mpirun.
1 parameter is path to data.
2 parameter is name of result file.

Example of running on 2 nodes specified in machine file named nodes:
mpirun -np 2 -machinefile nodes /home/3xcsokal/pso-mpi/program.exe /home/3xcsokal/data/iris.data result.txt