#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double greedy_tree_loglikelihood(int *matrix, int *node_genotypes, int cells, int nodes_length, int mutation_number, double *alpha, double beta) {

  double maximum_likelihood = 0;

  double like_00 = log(1-beta);
  double like_10 = log(beta);
  double *like_01 = malloc(mutation_number * sizeof(double));
  double *like_11 = malloc(mutation_number * sizeof(double));
  double like_2 = 0;

  for (int j = 0; j < mutation_number; j++) {
      like_01[j] = log(alpha[j]);
      like_11[j] = log(1 - alpha[j]);
  }

  for (int i = 0; i < cells; i++) {
      double best_lh = -9999999;

      for (int n = 0; n < nodes_length; n++) {
              double lh = 0;

              for (int j = 0; j < mutation_number; j++) {

                  double p = 1;

                  int I = matrix[i*mutation_number+j];
                  int E = node_genotypes[n*mutation_number+j];

                  if (I == 0 && E == 0) {
                      p = like_00;
                  } else if (I == 0 && E == 1) {
                      p = like_01[j];
                  } else if (I == 1 && E == 0) {
                      p = like_10;
                  } else if (I == 1 && E == 1) {
                      p = like_11[j];
                  } else if (I == 2) {
                      p = like_2;
                  }

                  lh += p;
              }

              if (lh > best_lh) {
                  best_lh = lh;
              }
      }
      maximum_likelihood += best_lh;
  }
  free(like_01);
  free(like_11);
  return maximum_likelihood;
}
