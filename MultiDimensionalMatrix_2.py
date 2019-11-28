import numpy as np
import pandas as pd
import os

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class NdMatrixManipulation(object):
    def read_gbm_parameters(self):
        # file_name_vol_corr = os.path.join(dir_path, 'data_outputs', 'vol_corr.csv')
        # matrix_vc = np.genfromtxt(file_name_vol_corr,  delimiter=',')
        # np.savetxt('matrix_vc.csv', matrix_vc, delimiter=',')
        matrix_vc = np.genfromtxt('matrix_vc.csv', delimiter=',')
        list_standard_deviation = np.sqrt(np.diag(matrix_vc))
        list_standard_deviation_inverse = np.power(list_standard_deviation, -1)

        matrix_diag_standard_deviation_inverse = np.diag(list_standard_deviation_inverse)
        matrix_correlation = np.linalg.multi_dot(
            [matrix_diag_standard_deviation_inverse,matrix_vc, matrix_diag_standard_deviation_inverse])
        # np.savetxt('matrix_corelation.csv', matrix_correlation, delimiter=',')
        correlation_chol_decompose_upper_triangle = np.linalg.cholesky(matrix_correlation).transpose()
        return correlation_chol_decompose_upper_triangle, list_standard_deviation

    def central_procedure(self):
        correlation_chol_decompose_upper_triangle, list_standard_deviation = self.read_gbm_parameters()
        matrix_random_normal_m_pricepath_n_riskfactors = self.generate_matrix_random_normal_m_pricepath_n_riskfactors()
        matrix_correlated_random_normal_m_pricepath_n_riskfactors = \
            self.generate_correlated_random_normal_m_pricepath_n_riskfactors(
                matrix_random_normal_m_pricepath_n_riskfactors, correlation_chol_decompose_upper_triangle)

        self.flatten_3d_matrix(matrix_correlated_random_normal_m_pricepath_n_riskfactors)

        # np.savetxt('random_normal_1_pricepath_n_riskfactors.csv', matrix_random_normal_1_pricepath_n_riskfactors,
        #            delimiter=',')
        # np.savetxt('correlated_random_normal_1_pricepath_n_riskfactors.csv',
        #            matrix_correlated_random_normal_1_pricepath_n_riskfactors, delimiter=',')

        return matrix_correlated_random_normal_m_pricepath_n_riskfactors
        # return correlation_chol_decompose_upper_triangle, list_standard_deviation

    def flatten_3d_matrix(self,nd_matrix):
        for index, data in enumerate(nd_matrix):
            if index == 0:
                flat_data = np.insert(data, 0, values=index, axis=1)
            else:
                data = np.insert(data, 0, values=index, axis=1)
                flat_data = np.vstack((flat_data, data))
        np.savetxt('three_dim_flattened.csv',
                   flat_data, delimiter=',')

        return None

    # generates matrix of normally distributed random numbers, mean=0, stdev = 1, columns=riskfactor, rows=tenor
    def generate_matrix_random_normal_m_pricepath_n_riskfactors(
            self,risk_factor_count=3, tenor_count=46, price_path_count=10, fixed_seed=10):
        np.random.seed(fixed_seed)
        matrix_random_normal_m_pricepath_n_riskfactors = \
            np.random.normal(size=(price_path_count,tenor_count, risk_factor_count))
        return matrix_random_normal_m_pricepath_n_riskfactors

    def generate_correlated_random_normal_m_pricepath_n_riskfactors(
            self, matrix_random_normal_m_pricepath_n_riskfactors, upper_triangular_matrix):

        # Here we iterate over each price path, send 2d matrix of tenor*RF to replace with correlated random normals
        for index, matrix2d_pricepath in enumerate(matrix_random_normal_m_pricepath_n_riskfactors):
            matrix_random_normal_m_pricepath_n_riskfactors[index] = \
                self.generate_correlated_random_normal_1_pricepath_n_riskfactors(
                    matrix_random_normal_m_pricepath_n_riskfactors[index], upper_triangular_matrix)

        return matrix_random_normal_m_pricepath_n_riskfactors

    def generate_correlated_random_normal_1_pricepath_n_riskfactors(
            self, matrix_random_normal_1_pricepath_n_riskfactors, upper_triangular_matrix):
        correlated_random_normal_1_pricepath_n_riskfactors = np.dot(matrix_random_normal_1_pricepath_n_riskfactors,
                                                                    upper_triangular_matrix)
        return correlated_random_normal_1_pricepath_n_riskfactors

    ### Defintion: S_t = S_0 * np.exp( ( mu - 0.5 * sigma**2) + sigma * W_t)
    def calc_1_day_gbm_change_vector(self,
                                     vector_sigma=np.array([0.02, 0.03, 0.04]),
                                     vector_W_t = np.array([1.33E+00, 7.38E-01, -1.26E+00])):
        vector_sigma_square = np.power(vector_sigma, 2)
        vector_sigma_square_half_negative = vector_sigma_square * (-0.5)
        vector_sigma_mult_vector_W_t = vector_sigma * vector_W_t
        vector_S_t = np.exp( vector_sigma_square_half_negative + vector_sigma_mult_vector_W_t )
        return vector_S_t

    ### Defintion: S_t = S_0 * np.exp( ( mu - 0.5 * sigma**2) + sigma * W_t)
    def calc_1_day_gbm_change_matrix(self):
        vector_sigma = np.array([0.02, 0.03, 0.04])
        vector_sigma_square = np.power(vector_sigma, 2)
        vector_sigma_square_half_negative = vector_sigma_square * (-0.5)

        matrix_diffusion = np.genfromtxt('matrix_diffusion.csv', delimiter=',')
        matrix_diffusion_times_sigma = vector_sigma * matrix_diffusion
        matrix_diffusion_times_sigma_negative_drift = matrix_diffusion_times_sigma + vector_sigma_square_half_negative
        matrix_percentile_multiplier_term = np.exp(matrix_diffusion_times_sigma_negative_drift)

        # Here we chain all standalone daily changes to make a price path
        for index, row in enumerate(matrix_percentile_multiplier_term):
            if index > 0:
                matrix_percentile_multiplier_term[index] = \
                    matrix_percentile_multiplier_term[index] * matrix_percentile_multiplier_term[index-1]

        return None


if __name__ == "__main__":
    print(NdMatrixManipulation().central_procedure())
    # SimulationGbm().generate_correlated_random_variables()
    # print(a)
