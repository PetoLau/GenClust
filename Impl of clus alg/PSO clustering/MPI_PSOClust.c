#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <math.h>

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

#define ROOT 0

typedef struct {
	double fitness;
	double local_best_fitness;
	double velocity[KMEANS][DIMENSIONS];
	double centroids[KMEANS][DIMENSIONS];
	double local_best_centroids[KMEANS][DIMENSIONS];
} PARTICLE;

//global variables
double vectors[DATA_ROWS][DIMENSIONS]; //data
char dataset[100];

PARTICLE particles[SWARM_SIZE];
PARTICLE mpi_particles[SWARM_SIZE];
PARTICLE global_best;

//PSO algorithm parameters
double w = 0.4;
double c1 = 2;
double c2 = 2;

/*
 random_int - return random_int from range min_num to max_num
 */
int random_int(int min_num, int max_num) {
	int result = 0;
	result = (rand() % (max_num - min_num)) + min_num;
	return result;
}

/*
 random_double - return random double between 0 and 1
 */
double random_double() {
	double result = 0;
	result = (double) rand() / (double) RAND_MAX;
	return result;
}

double my_abs(double x) {
	if (x >= 0.0) {
		return x;
	} else {
		return ((-1.0) * x);
	}
}

double user_defined_euclidean_distance(int particle, int centroid, int vector) {
#if USE_ROOTED_EUCLIDEAN_DISTANCE
	double size = 0;
	double psize = 0;
	int i;

	for (i = 0; i < DIMENSIONS; i++) {
		psize = vectors[vector][i] - mpi_particles[particle].centroids[centroid][i];
		size += psize * psize;
	}
	return (size);
#endif

#if USE_EUCLIDEAN_DISTANCE
	double size = 0;
	double psize = 0;
	int i;

	for (i = 0; i < DIMENSIONS; i++)
	{
		psize = vectors[vector][i] - mpi_particles[particle].centroids[centroid][i];
		size += psize * psize;
	}
	return sqrt(size);
#endif

#if USE_MANHATTAN_DISTANCE
	double size = 0;
	double psize = 0;
	int i;

	for (i = 0; i < DIMENSIONS; i++)
	{
		psize = vectors[vector][i] - mpi_particles[particle].centroids[centroid][i];
		size += my_abs(psize);
	}
	return (size);
#endif
}

void update_local_best(int particle) {
	int i, j;
	//updates local bests
	if (mpi_particles[particle].fitness < mpi_particles[particle].local_best_fitness) {
		mpi_particles[particle].local_best_fitness = mpi_particles[particle].fitness;
		for (i = 0; i < KMEANS; i++) {
			for (j = 0; j < DIMENSIONS; j++) {
				mpi_particles[particle].local_best_centroids[i][j] = mpi_particles[particle].centroids[i][j];
			}
		}
	}
}

int find_closest_centroid(double distances[DATA_ROWS][KMEANS], int vector) {
	int closest = 0;
	int i;
	for (i = 0; i < KMEANS; i++) {
		if (distances[vector][i] < distances[vector][closest]) {

			closest = i;
		}
	}
	return closest;
}

void calculate_fitness(int particle, double distances[DATA_ROWS][KMEANS]) {
	//local variables
	int vectorsFrequency[KMEANS];
	int closestCentroid[DATA_ROWS];
	double sum[KMEANS];
	int i;

	for (i = 0; i < KMEANS; i++) {
		sum[i] = 0;
		vectorsFrequency[i] = 0;
	}

	//find closest centroid to all data vectors
	for (i = 0; i < DATA_ROWS; i++) {
		closestCentroid[i] = find_closest_centroid(distances, i);
		vectorsFrequency[closestCentroid[i]]++;
	}

	//sum distance to closest centroids
	for (i = 0; i < DATA_ROWS; i++) {
		sum[closestCentroid[i]] += distances[i][closestCentroid[i]];
	}

	//divide sums by frequency
	for (i = 0; i < KMEANS; i++) {
		if (vectorsFrequency[i] != 0) {
			sum[i] = sum[i] / (double) vectorsFrequency[i];
		}
	}

	//sum and save fitness
	mpi_particles[particle].fitness = 0;
	for (i = 0; i < KMEANS; i++) {
		mpi_particles[particle].fitness += sum[i];
	}

	mpi_particles[particle].fitness = mpi_particles[particle].fitness / (double) KMEANS;

	//update local bests if applicable
	update_local_best(particle);
}

void UpdateVelocityAndPosition(int particle) {
	int i, j;

	//update velocity
	if (global_best.fitness == 10000.0) {
		for (i = 0; i < KMEANS; i++) {
			for (j = 0; j < DIMENSIONS; j++) {
				mpi_particles[particle].velocity[i][j] = w * mpi_particles[particle].velocity[i][j]
						+ c1 * random_double()
								* (mpi_particles[particle].local_best_centroids[i][j] - mpi_particles[particle].centroids[i][j]);
			}
		}
	} else {
		for (i = 0; i < KMEANS; i++) {
			for (j = 0; j < DIMENSIONS; j++) {
				mpi_particles[particle].velocity[i][j] = w * mpi_particles[particle].velocity[i][j]
						+ c1 * random_double()
								* (mpi_particles[particle].local_best_centroids[i][j] - mpi_particles[particle].centroids[i][j])
						+ c2 * random_double() * (global_best.centroids[i][j] - mpi_particles[particle].centroids[i][j]);
			}
		}
	}

	//update position
	for (i = 0; i < KMEANS; i++) {
		for (j = 0; j < DIMENSIONS; j++) {
			mpi_particles[particle].centroids[i][j] = mpi_particles[particle].centroids[i][j] + mpi_particles[particle].velocity[i][j];
		}
	}
}

void calculate(int i) {
	int k, l;
	double distances[DATA_ROWS][KMEANS];
	//calculate distances
	for (k = 0; k < DATA_ROWS; k++) {
		for (l = 0; l < KMEANS; l++) {
			distances[k][l] = user_defined_euclidean_distance(i, l, k);
		}
	}

	//calculate fitness
	calculate_fitness(i, distances);

	//calculate velocity and position
	UpdateVelocityAndPosition(i);
}

void update_global_best() {
	int i, j, k;
	for (i = 0; i < SWARM_SIZE; i++) {
		if (particles[i].local_best_fitness < global_best.fitness) {
			global_best.fitness = particles[i].local_best_fitness;
			for (j = 0; j < KMEANS; j++) {
				for (k = 0; k < DIMENSIONS; k++) {
					global_best.centroids[j][k] = particles[i].centroids[j][k];
				}
			}
		}
	}
}

void my_export(char filename[100]) {
	int i, k, l, closestCentroid[DATA_ROWS];
	double global_best_distances[DATA_ROWS][KMEANS];
	double size = 0;
	double psize = 0;

	for (k = 0; k < DATA_ROWS; k++) {
		for (l = 0; l < KMEANS; l++) {

#if USE_ROOTED_EUCLIDEAN_DISTANCE
			size = 0;
			psize = 0;

			for (i = 0; i < DIMENSIONS; i++) {
				psize = vectors[k][i] - global_best.centroids[l][i];
				size += psize * psize;
			}
			global_best_distances[k][l] = (size);
#endif
#if USE_EUCLIDEAN_DISTANCE
			size = 0;
			psize = 0;

			for (i = 0; i < DIMENSIONS; i++)
			{
				psize = vectors[k][i] - global_best.centroids[l][i];
				size += psize * psize;
			}
			global_best_distances[k][l] = sqrt(size);
#endif
#if USE_MANHATTAN_DISTANCE
			size = 0;
			psize = 0;

			for (i = 0; i < DIMENSIONS; i++)
			{
				psize = vectors[k][i] - global_best.centroids[l][i];
				size += abs(psize);
			}
			global_best_distances[k][l] = (size);
#endif
		}
	}

	for (k = 0; k < DATA_ROWS; k++) {
		closestCentroid[k] = find_closest_centroid(global_best_distances, k);
	}
	FILE *fp;
	fp = fopen(filename, "w");
	for (i = 0; i < DATA_ROWS; i++) {
		fprintf(fp, "%d\n", closestCentroid[i]);
	}
	fclose(fp);
}

int main(int argc, char *argv[]) {
	int i, j, k, l;
	time_t t, tt;

	int solver, solvers_count;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &solver);
	MPI_Comm_size(MPI_COMM_WORLD, &solvers_count);

	if (SWARM_SIZE % 2 == 1) {
		printf("SWARM SIZE need to be dividable/partible by 2\n");
		exit(1);
	}

	if (SWARM_SIZE % solvers_count != 0) {
		printf("SWARM_SIZE need to be divadable/partible by count of mpi solvers\n");
		exit(1);
	}

	if (argc >= 2) {
		strcpy(dataset, argv[1]);
	} else {
		printf("Program argument (dataset to read) not found.\n");
		exit(1);
	}

	srand(time(NULL));
	omp_set_num_threads(THREAD_COUNT);

	if (solver == ROOT) {
		for (i = 0; i < SWARM_SIZE; i++) {
			particles[i].fitness = 10000.0;
			particles[i].local_best_fitness = 10000.0;
		}
	}
	global_best.fitness = 10000.0;

	//read data
	FILE *fp;
	char *token;
	fp = fopen(dataset, "r");

	if (fp != NULL) {
		int lineNumber = 0;
		char line[256];
		while (fgets(line, sizeof line, fp) != NULL) {
			token = strtok(line, ",");
			for (i = 0; i < DIMENSIONS - 1; i++) {
				vectors[lineNumber][i] = atof(token);
				token = strtok(NULL, ",");
			}
			vectors[lineNumber][DIMENSIONS - 1] = atof(token);
			lineNumber++;
		}
		fclose(fp);
	} else {
		printf("File read problem. Check correct name and file location/position.\n");
		exit(1);
	}
	//end of read data

	time(&t);

	//initialize swarm
	if (solver == ROOT) {
#pragma omp parallel for private(j,k,l)
		for (i = 0; i < SWARM_SIZE; i++) {
			for (j = 0; j < KMEANS; j++) {
				k = random_int(0, DATA_ROWS);
				for (l = 0; l < DIMENSIONS; l++) {
					particles[i].centroids[j][l] = vectors[k][l];
					particles[i].velocity[j][l] = random_double() * 6.0 - 3.0;
				}
			}
		}
	}
	//end of swarm initialization

	//mpi particle struct code
	int struct_count = 5;
	int struct_lengths[5] = { 1, 1, KMEANS * DIMENSIONS, KMEANS * DIMENSIONS,
	KMEANS * DIMENSIONS };
	MPI_Aint struct_offsets[5] = { 0, sizeof(double), 2 * sizeof(double), 2 * sizeof(double) + sizeof(double) * KMEANS * DIMENSIONS, 2
			* sizeof(double) + 2 * sizeof(double) * KMEANS * DIMENSIONS };
	MPI_Datatype struct_types[5] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
	MPI_DOUBLE, MPI_DOUBLE };

	MPI_Datatype myTypeParticle;
	MPI_Type_struct(struct_count, struct_lengths, struct_offsets, struct_types, &myTypeParticle);
	MPI_Type_commit(&myTypeParticle);

	//main loop
	for (i = 0; i < MAX_ITERATIONS; i++) {
		//send particles
		MPI_Scatter(particles,
		SWARM_SIZE / solvers_count, myTypeParticle, mpi_particles,
		SWARM_SIZE / solvers_count, myTypeParticle,
		ROOT,
		MPI_COMM_WORLD);

		//send global best
		MPI_Bcast(&global_best, 1, myTypeParticle, ROOT, MPI_COMM_WORLD);

		//do work
#pragma omp parallel for shared(j)
		for (j = 0; j < SWARM_SIZE / solvers_count; j++) {
			calculate(j);
		}

		//gather particles
		MPI_Gather(mpi_particles,
		SWARM_SIZE / solvers_count, myTypeParticle, particles,
		SWARM_SIZE / solvers_count, myTypeParticle,
		ROOT,
		MPI_COMM_WORLD);

		//find global best if applicable
		if (solver == ROOT) {
			update_global_best();
		}

	}
	//end of main loop

	MPI_Finalize();

	if (solver == ROOT) {
		time(&tt);
		printf("Program bezal %.f sekund.\n", difftime(tt, t));

		printf("Najlepsie dosiahnuta fitness: %lf\n", global_best.fitness);
		printf("USE_ROOTED_EUCLIDEAN_DISTANCE %d\n", USE_ROOTED_EUCLIDEAN_DISTANCE);
		printf("USE_EUCLIDEAN_DISTANCE %d\n", USE_EUCLIDEAN_DISTANCE);
		printf("USE_MANHATTAN_DISTANCE %d\n", USE_MANHATTAN_DISTANCE);
		printf("w: %lf\n", w);
		printf("c1: %lf\n", c1);
		printf("c2: %lf\n", c2);
		printf("SWARM SIZE: %d\n", SWARM_SIZE);
		printf("MAX_ITERATIONS: %d\n", MAX_ITERATIONS);
		printf("DIMENSIONS: %d\n", DIMENSIONS);
		printf("KMEANS: %d\n", KMEANS);
		printf("THREAD_COUNT per instance: %d\n", THREAD_COUNT);
		printf("mpi solvers_count: %d\n", solvers_count);

		if (argc >= 3) {
			my_export(argv[2]);
		} else {
			my_export("/home/lukas/workspace_parallel/pso_mpi/export2.txt");
		}
	}

	return 0;
}
