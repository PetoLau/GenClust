Requirments to compile and run:
OpenMPI or Mpich
GNU GSL lib (https://www.gnu.org/software/gsl/)
GCC
--------------------------------------------
Suggest we installed GSL in folder /home/sample_user/gsl-libs/lib , then commands to compile are:

LD_LIBRARY_PATH=/home/sample_user/gsl-libs/lib
export LD_LIBRARY_PATH
mpicc MPI_GenClust.c -o3 -fopenmp -I/home/sample_user/gsl-libs/include/ -L/home/sample_user/gsl-libs/lib -lm -Wl,-Bstatic -lgsl -lgslcblas -Wl,-Bdynamic -o program.exe
--------------------------------------------
Commands to compile, when GSL and other libs are located in standard folder:

mpicc MPI_GenClust.c -o3 -fopenmp -o program.exe
--------------------------------------------
Parameters are located on the beggining of source code, example of parameters to cluster IRIS:
//data specific parameters
#define KMEANS 3 //number of cluster groups
#define DIMENSIONS 4 //number of dimensions of data
#define DATA_ROWS 150

//run specific parameters
#define THREAD_COUNT 6 //count of thread used by OpenMP
#define FITNESS_FUNCTION fitness5 //specify fitness function used, 5 possibilities (fitness1,fitness2,...,fitness5)
#define SWARM_SIZE (200)
#define MAX_ITERATIONS (10000)
#define GEN_MUTATION 4 //how many parts of genes will be mutated
#define ELITE 4 //how many genes will be preserved without change
--------------------------------------------
To cluster data, run compiled program with parameters. Use program mpirun.
1 parameter is path to data.
2 parameter is name of result file.

Example of running on 2 nodes specified in machine file named nodes:
mpirun -np 2 -machinefile nodes /home/sample_user/gen-mpi/program.exe /home/sample_user/data/iris.data result.txt
