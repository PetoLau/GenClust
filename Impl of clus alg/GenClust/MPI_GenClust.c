#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permute_vector.h>
#include <gsl/gsl_blas.h>
#include "mpi.h"

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

#define REAL long double
#define size_t int
#define ROOT 0

long double vectors[DATA_ROWS][DIMENSIONS]; //here are stored data vectors

long double fitness[SWARM_SIZE];
int chrom[SWARM_SIZE][DATA_ROWS];
int new_chromosoms[SWARM_SIZE][DATA_ROWS];

long double fit_data_vectors[KMEANS][DIMENSIONS][DATA_ROWS]; //used to compute fitness
#pragma omp threadprivate(fit_data_vectors)
char dataset[100];

/*
 * FITNEES FUNCTION
 * fitness1 - vseobecne kriterium (VVV)
 * fitness2 - spriemerovane kovariancne matice sa pouziju na fitness (EEE)
 * fitness3 - stopa kovariancnych matic (VII)
 * fitness4 - sucet stvorcov euklidovskych vzdialenosti fitness (KMEANS)
 * fitness5 - spriemerovane stopy kovariancnych matic (EII)
 */

long double my_mean(const long double data[], const size_t stride,
		const size_t size) {
	long double mean = 0;
	size_t i;

	for (i = 0; i < size; i++) {
		mean += (data[i * stride] - mean) / (i + 1);
	}

	return mean;
}

long double _my_covariance(const long double data1[], const size_t stride1,
		const long double data2[], const size_t stride2, const size_t n,
		const long double mean1, const long double mean2) {

	long double covariance = 0;

	size_t i;

	/* find the sum of the squares */
	for (i = 0; i < n; i++) {
		const long double delta1 = (data1[i * stride1] - mean1);
		const long double delta2 = (data2[i * stride2] - mean2);
		covariance += (delta1 * delta2 - covariance) / (i + 1);
	}

	return covariance;
}

long double my_gsl_stats_covariance(const long double data1[],
		const size_t stride1, const long double data2[], const size_t stride2,
		const size_t n) {
	const long double mean1 = my_mean(data1, stride1, n);
	const long double mean2 = my_mean(data2, stride2, n);

	return _my_covariance(data1, stride1, data2, stride2, n, mean1, mean2);
}

long double my_gsl_linalg_LU_det(gsl_matrix_long_double * LU, int signum) {
	size_t i, n = LU->size1;
	long double det = (long double) signum;

	for (i = 0; i < n; i++) {
		det *= gsl_matrix_long_double_get(LU, i, i);
	}

	return det;
}

void my_gsl_linalg_LU_decomp(gsl_matrix_long_double * A, gsl_permutation * p,
		int *signum) {
	if (A->size1 != A->size2) {
		printf("LU decomposition requires square matrix");
	} else if (p->size != A->size1) {
		printf("permutation length must match matrix size");
	} else {
		const size_t N = A->size1;
		size_t i, j, k;

		*signum = 1;
		gsl_permutation_init(p);

		for (j = 0; j < N - 1; j++) {
			/* Find maximum in the j-th column */

			long double ajj, max = fabs(gsl_matrix_long_double_get(A, j, j));
			size_t i_pivot = j;

			for (i = j + 1; i < N; i++) {
				long double aij = fabs(gsl_matrix_long_double_get(A, i, j));

				if (aij > max) {
					max = aij;
					i_pivot = i;
				}
			}

			if (i_pivot != j) {
				gsl_matrix_long_double_swap_rows(A, j, i_pivot);
				gsl_permutation_swap(p, j, i_pivot);
				*signum = -(*signum);
			}

			ajj = gsl_matrix_long_double_get(A, j, j);

			if (ajj != 0.0) {
				for (i = j + 1; i < N; i++) {
					long double aij = gsl_matrix_long_double_get(A, i, j) / ajj;
					gsl_matrix_long_double_set(A, i, j, aij);

					for (k = j + 1; k < N; k++) {
						long double aik = gsl_matrix_long_double_get(A, i, k);
						long double ajk = gsl_matrix_long_double_get(A, j, k);
						gsl_matrix_long_double_set(A, i, k, aik - aij * ajk);
					}
				}
			}
		}
	}
}

int random_int(int min_num, int max_num) {
	int result = 0;
	result = (rand() % (max_num - min_num)) + min_num;
	return result;
}

int my_ceil(float num) {
	int inum = (int) num;
	if (num == (float) inum) {
		return inum;
	}
	return inum + 1;
}

void my_export(char filename[100]) {
	int i;
	FILE *fp;
	fp = fopen(filename, "w");
	for (i = 0; i < DATA_ROWS; i++) {
		fprintf(fp, "%d\n", chrom[0][i] + 1);
	}
	fclose(fp);
}

long double fitness1(int i) {
	int j, k, h, s;
	long double det;
	int fit_cluster_volume[KMEANS];
	gsl_matrix_long_double *fit_matrix[KMEANS];
	gsl_permutation * p = gsl_permutation_alloc(DIMENSIONS);
	long double result = 0;

	for (j = 0; j < KMEANS; j++) {
		fit_cluster_volume[j] = 0;
	}

	for (j = 0; j < DATA_ROWS; j++) {
		for (h = 0; h < DIMENSIONS; h++) {
			fit_data_vectors[chrom[i][j]][h][fit_cluster_volume[chrom[i][j]]] =
					vectors[j][h];
		}

		fit_cluster_volume[chrom[i][j]]++;
	}

	for (j = 0; j < KMEANS; j++) {
		fit_matrix[j] = gsl_matrix_long_double_calloc(DIMENSIONS, DIMENSIONS);
		for (k = 0; k < DIMENSIONS; k++) {
			for (h = 0; h < DIMENSIONS; h++) {
				gsl_matrix_long_double_set(fit_matrix[j], k, h,
						my_gsl_stats_covariance(fit_data_vectors[j][k], 1,
								fit_data_vectors[j][h], 1,
								fit_cluster_volume[j]));
			}
		}
	}

	for (j = 0; j < KMEANS; j++) {
		s = 1;
		my_gsl_linalg_LU_decomp(fit_matrix[j], p, &s);
		det = my_gsl_linalg_LU_det(fit_matrix[j], s);
		if (det != 0.0) {
			result += gsl_sf_log_abs(det) * (double) fit_cluster_volume[j];
		}
		gsl_matrix_long_double_free(fit_matrix[j]);
	}
	return result;
}

long double fitness2(int i) {
	int j, k, h, s;
	long double m, det;
	int fit_cluster_volume[KMEANS];
	gsl_matrix_long_double *fit_matrix[KMEANS];
	gsl_matrix_long_double *fit_reuslt_matrix;
	gsl_permutation * p = gsl_permutation_alloc(DIMENSIONS);
	long double result = 0;

	for (j = 0; j < KMEANS; j++) {
		fit_cluster_volume[j] = 0;
	}

	for (j = 0; j < DATA_ROWS; j++) {
		for (h = 0; h < DIMENSIONS; h++) {
			fit_data_vectors[chrom[i][j]][h][fit_cluster_volume[chrom[i][j]]] =
					vectors[j][h];
		}

		fit_cluster_volume[chrom[i][j]]++;
	}

	for (j = 0; j < KMEANS; j++) {
		fit_matrix[j] = gsl_matrix_long_double_calloc(DIMENSIONS, DIMENSIONS);
		for (k = 0; k < DIMENSIONS; k++) {
			for (h = 0; h < DIMENSIONS; h++) {
				gsl_matrix_long_double_set(fit_matrix[j], k, h,
						my_gsl_stats_covariance(fit_data_vectors[j][k], 1,
								fit_data_vectors[j][h], 1,
								fit_cluster_volume[j]));
			}
		}
	}

	fit_reuslt_matrix = gsl_matrix_long_double_calloc(DIMENSIONS, DIMENSIONS);

	for (j = 0; j < KMEANS; j++) {
		for (k = 0; k < DIMENSIONS; k++) {
			for (h = 0; h < DIMENSIONS; h++) {

				m = gsl_matrix_long_double_get(fit_reuslt_matrix, k, h);
				m += fit_cluster_volume[j]
						* gsl_matrix_long_double_get(fit_matrix[j], k, h);
				gsl_matrix_long_double_set(fit_reuslt_matrix, k, h, m);
			}
		}
		gsl_matrix_long_double_free(fit_matrix[j]);
	}

	for (k = 0; k < DIMENSIONS; k++) {
		for (h = 0; h < DIMENSIONS; h++) {
			m = gsl_matrix_long_double_get(fit_reuslt_matrix, k, h);
			m = m / (double) DATA_ROWS;
			gsl_matrix_long_double_set(fit_reuslt_matrix, k, h, m);
		}
	}

	s = 1;
	my_gsl_linalg_LU_decomp(fit_reuslt_matrix, p, &s);
	det = my_gsl_linalg_LU_det(fit_reuslt_matrix, s);
	if (det != 0.0) {
		result += gsl_sf_log_abs(det);
	}
	gsl_matrix_long_double_free(fit_reuslt_matrix);
	return result;
}

long double fitness3(int i) {
	int j, k, h;
	int fit_cluster_volume[KMEANS];
	gsl_matrix_long_double *fit_matrix[KMEANS];
	long double result = 0;

	for (j = 0; j < KMEANS; j++) {
		fit_cluster_volume[j] = 0;
	}

	for (j = 0; j < DATA_ROWS; j++) {
		for (h = 0; h < DIMENSIONS; h++) {
			fit_data_vectors[chrom[i][j]][h][fit_cluster_volume[chrom[i][j]]] =
					vectors[j][h];
		}

		fit_cluster_volume[chrom[i][j]]++;
	}

	for (j = 0; j < KMEANS; j++) {
		fit_matrix[j] = gsl_matrix_long_double_calloc(DIMENSIONS, DIMENSIONS);
		for (k = 0; k < DIMENSIONS; k++) {
			for (h = 0; h < DIMENSIONS; h++) {
				gsl_matrix_long_double_set(fit_matrix[j], k, h,
						my_gsl_stats_covariance(fit_data_vectors[j][k], 1,
								fit_data_vectors[j][h], 1,
								fit_cluster_volume[j]));
			}
		}
	}

	for (j = 0; j < KMEANS; j++) {
		for (k = 0; k < DIMENSIONS; k++) {
			result += gsl_matrix_long_double_get(fit_matrix[j], k, k);
		}
		gsl_matrix_long_double_free(fit_matrix[j]);
	}
	return result;
}

long double fitness4(int agent) {
	int i, j, v;
	int weight_matrix[DATA_ROWS][KMEANS];
	long double center_matrix[KMEANS][DIMENSIONS];
	long double sum_w;
	long double sum_wx;
	long double result = 0;

	for (i = 0; i < DATA_ROWS; i++) {
		for (j = 0; j < KMEANS; j++) {
			if (chrom[agent][i] == j) {
				weight_matrix[i][j] = 1;
			} else {
				weight_matrix[i][j] = 0;
			}
		}
	}

	for (j = 0; j < KMEANS; j++) {
		for (v = 0; v < DIMENSIONS; v++) {
			sum_w = 0.0;
			for (i = 0; i < DATA_ROWS; i++) {
				sum_w += weight_matrix[i][j];
			}

			sum_wx = 0.0;
			for (i = 0; i < DATA_ROWS; i++) {
				sum_wx += weight_matrix[i][j] * vectors[i][v];
			}

			center_matrix[j][v] = sum_wx / sum_w;
		}
	}

	for (j = 0; j < KMEANS; j++) {
		for (i = 0; i < DATA_ROWS; i++) {
			sum_w = 0.0;
			for (v = 0; v < DIMENSIONS; v++) {
				sum_w += weight_matrix[i][j]
						* ((vectors[i][v] - center_matrix[j][v])
								* (vectors[i][v] - center_matrix[j][v]));
			}
			result += sqrt(sum_w);

		}
	}
	return result;
}

long double fitness5(int i) {
	int j, k, h;
	long double m;
	int fit_cluster_volume[KMEANS];
	gsl_matrix_long_double *fit_matrix[KMEANS];
	gsl_matrix_long_double *fit_reuslt_matrix;
	long double result = 0;

	for (j = 0; j < KMEANS; j++) {
		fit_cluster_volume[j] = 0;
	}

	for (j = 0; j < DATA_ROWS; j++) {
		for (h = 0; h < DIMENSIONS; h++) {
			fit_data_vectors[chrom[i][j]][h][fit_cluster_volume[chrom[i][j]]] =
					vectors[j][h];
		}

		fit_cluster_volume[chrom[i][j]]++;
	}

	for (j = 0; j < KMEANS; j++) {
		fit_matrix[j] = gsl_matrix_long_double_calloc(DIMENSIONS, DIMENSIONS);
		for (k = 0; k < DIMENSIONS; k++) {
			for (h = 0; h < DIMENSIONS; h++) {
				gsl_matrix_long_double_set(fit_matrix[j], k, h,
						my_gsl_stats_covariance(fit_data_vectors[j][k], 1,
								fit_data_vectors[j][h], 1,
								fit_cluster_volume[j]));
			}
		}
	}

	fit_reuslt_matrix = gsl_matrix_long_double_calloc(DIMENSIONS, DIMENSIONS);

	for (j = 0; j < KMEANS; j++) {
		for (k = 0; k < DIMENSIONS; k++) {
			for (h = 0; h < DIMENSIONS; h++) {

				m = gsl_matrix_long_double_get(fit_reuslt_matrix, k, h);
				m += fit_cluster_volume[j]
						* gsl_matrix_long_double_get(fit_matrix[j], k, h);
				gsl_matrix_long_double_set(fit_reuslt_matrix, k, h, m);
			}
		}
		gsl_matrix_long_double_free(fit_matrix[j]);
	}

	for (k = 0; k < DIMENSIONS; k++) {
		for (h = 0; h < DIMENSIONS; h++) {
			m = gsl_matrix_long_double_get(fit_reuslt_matrix, k, h);
			m = m / (double) DATA_ROWS;
			gsl_matrix_long_double_set(fit_reuslt_matrix, k, h, m);
		}
	}

	for (k = 0; k < DIMENSIONS; k++) {
		result += gsl_matrix_long_double_get(fit_reuslt_matrix, k, k);
	}
	gsl_matrix_long_double_free(fit_reuslt_matrix);

	return result;
}

void quicksort(long double list[], int ch[][DATA_ROWS], int n) {
	int i, j;
	long double pivot;
	long double temp;
	int temp_ch[DATA_ROWS];

	if (n < 2)
		return;
	pivot = list[n / 2];
	for (i = 0, j = n - 1;; i++, j--) {
		while (list[i] < pivot)
			i++;
		while (pivot < list[j])
			j--;
		if (i >= j)
			break;
		//swap
		temp = list[i];
		list[i] = list[j];
		list[j] = temp;
		memcpy(&temp_ch, &ch[i][0], sizeof(int) * DATA_ROWS);
		memcpy(&ch[i][0], &ch[j][0], sizeof(int) * DATA_ROWS);
		memcpy(&ch[j][0], &temp_ch, sizeof(int) * DATA_ROWS);

	}
	quicksort(list, ch, i);
	quicksort(list + i, ch + i, n - i);
}

int *int_array_2dto1d(int array[][DATA_ROWS], int first_dimension,
		int second_dimension) {
	int i, j;
	int *subarray = malloc(sizeof(int) * first_dimension * second_dimension);
	for (i = 0; i < first_dimension; i++) {
		for (j = 0; j < second_dimension; j++) {
			subarray[i * second_dimension + j] = array[i][j];
		}
	}
	return subarray;
}

void int_array_1dto2d(int *array, int solver, int elements_per_proc,
		int first_dimension, int second_dimension) {
	int i, j;
	for (i = 0; i < first_dimension; i++) {
		for (j = 0; j < second_dimension; j++) {
			chrom[solver * elements_per_proc + i][j] = array[i
					* second_dimension + j];
		}
	}
}

int main(int argc, char *argv[]) {
	int i, j, k, h, it;

	int temp_ch[DATA_ROWS];
	int *sub_chrom;
	int * chrom1d = NULL;
	long double *sub_fitness;

	unsigned long long int mating_h[SWARM_SIZE];
	unsigned long long int max_mating = 0;
	int mates[SWARM_SIZE];

	int r1, r2;

	time_t t, tt;

	int solver, solvers_count;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &solver);
	MPI_Comm_size(MPI_COMM_WORLD, &solvers_count);

	MPI_Datatype my_type_scatter;
	MPI_Type_contiguous(DATA_ROWS, MPI_INT, &my_type_scatter);
	MPI_Type_commit(&my_type_scatter);

	if (SWARM_SIZE % 2 == 1) {
		printf("SWARM SIZE need to be dividable/partible by 2\n");
		exit(1);
	}

	if (argc >= 2) {
		strcpy(dataset, argv[1]);
	} else {
		printf("Program argument (dataset to read) not found.\n");
		exit(1);
	}

	if (SWARM_SIZE % solvers_count != 0) {
		printf(
				"SWARM_SIZE need to be divadable/partible by count of mpi solvers\n");
		exit(1);
	}

	srand(time(NULL));
	omp_set_num_threads(THREAD_COUNT);

	if (solver == ROOT) {
		for (i = 0; i < SWARM_SIZE / 2; i++) {
			mating_h[i] = gsl_pow_int((SWARM_SIZE / 2 - i), 5);
			max_mating += mating_h[i];
			mating_h[i + SWARM_SIZE / 2] = 0;
		}
	}

//read data
	FILE *fp;
	char *token;
	fp = fopen(dataset, "r");

	if (fp != NULL) {
		int lineNumber = 0;
		char line[1024];
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
		printf("%s\n", strerror(errno));
		exit(1);
	}
//end of read data

	time(&t);

//Generate initial population.
	if (solver == ROOT) {
		for (i = 0; i < SWARM_SIZE; i++) {
			for (j = 0; j < DATA_ROWS; j++) {
				chrom[i][j] = random_int(0, KMEANS);
			}
		}
	}

	//Compute fitness of each chromosome.
#pragma omp parallel for shared(i,fitness)
	for (i = 0; i < SWARM_SIZE; i++) {
		fitness[i] = FITNESS_FUNCTION(i);
	}

	quicksort(fitness, chrom, SWARM_SIZE);

	for (it = 0; it < MAX_ITERATIONS; it++) {
		//Natural selection. Select mates.
		if (solver == ROOT) {
			for (i = 0; i < SWARM_SIZE; i++) {
				j = random_int(1, max_mating);
				h = 0;
				k = mating_h[h];
				while (j > k) {
					k += mating_h[++h];
				}
				mates[i] = h;
			}
		}

		//Mating
		if (solver == ROOT) {
			for (i = 0; i < SWARM_SIZE; i++) {
				do {
					r1 = random_int(0, DATA_ROWS);
					r2 = random_int(0, DATA_ROWS);
				} while (r1 == r2);

				if (r1 < r2) {
					j = r1;
					k = r2;
				} else {
					j = r2;
					k = r1;
				}

				//first kid
				memcpy(&temp_ch[0], &chrom[mates[i]][0], j * sizeof(int));
				memcpy(&temp_ch[j], &chrom[mates[i + 1]][j],
						(k - j) * sizeof(int));
				memcpy(&temp_ch[k], &chrom[mates[i]][k],
						(DATA_ROWS - k) * sizeof(int));
				memcpy(new_chromosoms[i], temp_ch, DATA_ROWS * sizeof(int));

				//second kid
				memcpy(&temp_ch[0], &chrom[mates[i + 1]][0], j * sizeof(int));
				memcpy(&temp_ch[j], &chrom[mates[i]][j], (k - j) * sizeof(int));
				memcpy(&temp_ch[k], &chrom[mates[i + 1]][k],
						(DATA_ROWS - k) * sizeof(int));
				memcpy(new_chromosoms[i + 1], temp_ch, DATA_ROWS * sizeof(int));
				i++;
			}
		}

		//copy new population
		if (solver == ROOT) {
			for (i = ELITE; i < SWARM_SIZE; i++) {
				memcpy(&chrom[i], &new_chromosoms[i], sizeof(int) * DATA_ROWS);
			}
		}

		//Mutation.
		if (solver == ROOT) {
			for (i = ELITE; i < SWARM_SIZE; i++) {
				for (j = 0; j < GEN_MUTATION; j++) {
					k = random_int(0, DATA_ROWS);
					chrom[i][k] = random_int(0, KMEANS);
				}
			}
		}

		//mpi section
		if (solver == ROOT) {
			chrom1d = int_array_2dto1d(chrom, SWARM_SIZE, DATA_ROWS);
		}

		sub_chrom = malloc(
				sizeof(int) * (SWARM_SIZE / solvers_count) * DATA_ROWS);


		MPI_Scatter(chrom1d,
		SWARM_SIZE / solvers_count, my_type_scatter, sub_chrom,
		SWARM_SIZE / solvers_count, my_type_scatter,
		ROOT,
		MPI_COMM_WORLD);

		int_array_1dto2d(sub_chrom, solver, SWARM_SIZE / solvers_count,
				SWARM_SIZE / solvers_count, DATA_ROWS);
		sub_fitness = malloc(
				sizeof(long double) * (SWARM_SIZE / solvers_count));


		//Compute fitness of each chromosome.
#pragma omp parallel for shared(i,fitness,sub_fitness)
		for (i = solver * (SWARM_SIZE / solvers_count);
				i
						< solver * (SWARM_SIZE / solvers_count)
								+ (SWARM_SIZE / solvers_count); i++) {
			sub_fitness[i - solver * (SWARM_SIZE / solvers_count)] =
					FITNESS_FUNCTION(i);
		}

		MPI_Gather(sub_fitness,
		SWARM_SIZE / solvers_count,
		MPI_LONG_DOUBLE, fitness,
		SWARM_SIZE / solvers_count,
		MPI_LONG_DOUBLE,
		ROOT,
		MPI_COMM_WORLD);


		free(sub_chrom);
		free(sub_fitness);
		if (solver == ROOT && chrom1d != NULL) {
			free(chrom1d);
		}
		//end of mpi section

		//Order population
		if (solver == ROOT) {
			quicksort(fitness, chrom, SWARM_SIZE);
		}
	}

	if (solver == ROOT) {

		time(&tt);
		printf("Program bezal %.f sekund.\n", difftime(tt, t));

		printf("fitness: %Lf\n", fitness[0]);

		if (argc >= 3) {
			my_export(argv[2]);
		} else {
			my_export("/home/lukas/workspace_parallel/mpi_gen/export.txt");
		}

		printf("SWARM SIZE: %d\n", SWARM_SIZE);
		printf("MAX_ITERATIONS: %d\n", MAX_ITERATIONS);
		printf("GEN_MUTATION: %d\n", GEN_MUTATION);
		printf("ELITE: %d\n", ELITE);
		printf("DIMENSIONS: %d\n", DIMENSIONS);
		printf("KMEANS: %d\n", KMEANS);
		printf("THREAD_COUNT per instance: %d\n", THREAD_COUNT);
		printf("mpi solvers_count: %d\n", solvers_count);
		printf(" FITNESS_FUNCTION ");

	}

	MPI_Finalize();

	return 0;

}
