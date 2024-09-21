import numpy as np
from scipy.special import erf

def special_function(x): return np.where(x == 0, 1, (0.5 * (np.pi / (x + 1e-17)) ** 0.5) * erf((x + 1e-17) ** 0.5))


def evaluate_integrals(orbitals, Z_list, atomic_coords, centre_of_mass, two_electron_ints=True):

    nbasis = len(orbitals)

    S = np.zeros([nbasis, nbasis])
    T = np.zeros([nbasis, nbasis])
    V_NE = np.zeros([nbasis, nbasis])
    D = np.zeros([nbasis, nbasis])
    V_EE = np.zeros([nbasis, nbasis, nbasis, nbasis])

    for i in range(nbasis):
        for j in range(i, nbasis):  

            alphas_m = np.array([pg.alpha for pg in orbitals[i]])
            alphas_n = np.array([pg.alpha for pg in orbitals[j]])

            coeffs_m = np.array([pg.coeff for pg in orbitals[i]])
            coeffs_n = np.array([pg.coeff for pg in orbitals[j]])

            R_m = np.array([pg.coordinates for pg in orbitals[i]])
            R_n = np.array([pg.coordinates for pg in orbitals[j]])

            sum_mn = alphas_m[:, np.newaxis] + alphas_n
            product_mn = alphas_m[:, np.newaxis] * alphas_n
            coeffproduct_mn = coeffs_m[:, np.newaxis] * coeffs_n
            R_mn = np.linalg.norm(R_m[:, np.newaxis] - R_n, axis=2)

            R_m_com = R_m - np.array([0, 0, centre_of_mass])
            R_n_com = R_n - np.array([0, 0, centre_of_mass])
            
            alpha_m_R_m = np.einsum("i, ij->ij", alphas_m, R_m)
            alpha_n_R_n = np.einsum("i, ij->ij", alphas_n, R_n)

            alpha_m_R_m_com = np.einsum("i, ij->ij", alphas_m, R_m_com)
            alpha_n_R_n_com = np.einsum("i, ij->ij", alphas_n, R_n_com)
                    
            OM_mn = coeffproduct_mn * (4 * product_mn / sum_mn ** 2) ** (3 / 4) * np.exp(-(product_mn / sum_mn) * R_mn ** 2)

            Rk = np.einsum("ijk,ij->ij",(alpha_m_R_m[:, np.newaxis] + alpha_n_R_n), 1 / sum_mn)
            Rk_dipole = np.einsum("ijk,ij->ij", alpha_m_R_m_com[:, np.newaxis] + alpha_n_R_n_com, 1 / sum_mn)

            S[i,j] = np.einsum("mn->", OM_mn)
            T[i,j] = np.einsum("mn,mn,mn->", OM_mn, (product_mn / sum_mn), (3 - (2 * product_mn * R_mn**2) / sum_mn))
            D[i,j] += np.einsum("mn,mn->", OM_mn, Rk_dipole)

            for atom in range(len(Z_list)):

                dfunc_to_atom_mn = (Rk - atomic_coords[atom][2]) ** 2

                V_NE[i,j] += -Z_list[atom] * np.einsum("mn,mn,mn->",OM_mn,special_function(sum_mn * dfunc_to_atom_mn), 2 * np.sqrt(sum_mn / np.pi))
            
            S[j,i] = S[i,j]
            D[j,i] = D[i,j]
            T[j,i] = T[i,j]
            V_NE[j,i] = V_NE[i,j] 
                
            if two_electron_ints:

                for k in range(i, nbasis):
                    for l in range(k, nbasis): 

                        
                        alphas_o = np.array([pg.alpha for pg in orbitals[k]])
                        alphas_p = np.array([pg.alpha for pg in orbitals[l]])

                        coeffs_o = np.array([pg.coeff for pg in orbitals[k]])
                        coeffs_p = np.array([pg.coeff for pg in orbitals[l]])

                        R_o = np.array([pg.coordinates for pg in orbitals[k]])
                        R_p = np.array([pg.coordinates for pg in orbitals[l]])

                        sum_op = alphas_o[:, np.newaxis] + alphas_p
                        product_op = alphas_o[:, np.newaxis] * alphas_p
                        coeffproduct_op = coeffs_o[:, np.newaxis] * coeffs_p
                        R_op = np.linalg.norm(R_o[:, np.newaxis] - R_p, axis=2)

                        alpha_o_R_o = np.einsum("i, ij->ij", alphas_o, R_o)
                        alpha_p_R_p = np.einsum("i, ij->ij", alphas_p, R_p)

                        Rl = np.einsum("ijk,ij->ij",(alpha_o_R_o[:, np.newaxis] + alpha_p_R_p), 1 / sum_op)

                        OM_op = coeffproduct_op * (4 * product_op / sum_op ** 2) ** (3 / 4) * np.exp(-(product_op / sum_op) * R_op ** 2)
                        
                        prod_over_sum = np.einsum("mn,op,mnop->mnop",sum_mn, sum_op, 1 / (sum_mn[:, :, np.newaxis, np.newaxis] + sum_op[np.newaxis, np.newaxis, :, :]))

                        RkRl = (Rk[:, :, np.newaxis, np.newaxis] - Rl[np.newaxis, np.newaxis, :, :])**2

                        input_function = np.einsum("ijkl,ijkl->ijkl", prod_over_sum, RkRl)

                        V_EE[i,j,k,l] = 2 / np.sqrt(np.pi) * np.einsum("mnop,mnop,mn,op->",np.sqrt(prod_over_sum), special_function(input_function), OM_mn, OM_op)
                    
                    
                        V_EE[j, i, l, k] = V_EE[i,j,k,l]
                        V_EE[j, i, k, l] = V_EE[i,j,k,l]
                        V_EE[i, j, l, k] = V_EE[i,j,k,l]
                        V_EE[k, l, i, j] = V_EE[i,j,k,l]
                        V_EE[l, k, j, i] = V_EE[i,j,k,l]
                        V_EE[l, k, i, j] = V_EE[i,j,k,l]
                        V_EE[k, l, j, i] = V_EE[i,j,k,l]


    return S, T, V_NE, D, V_EE


def dipoe(orbitals, centre_of_mass):

    nbasis = len(orbitals)
    D = np.zeros([nbasis, nbasis])


    for i in range(nbasis):
        for j in range(i, nbasis):  

            alphas_m = np.array([pg.alpha for pg in orbitals[i]])
            alphas_n = np.array([pg.alpha for pg in orbitals[j]])

            coeffs_m = np.array([pg.coeff for pg in orbitals[i]])
            coeffs_n = np.array([pg.coeff for pg in orbitals[j]])

            R_m = np.array([pg.coordinates for pg in orbitals[i]])
            R_n = np.array([pg.coordinates for pg in orbitals[j]])

            sum_mn = alphas_m[:, np.newaxis] + alphas_n
            product_mn = alphas_m[:, np.newaxis] * alphas_n
            coeffproduct_mn = coeffs_m[:, np.newaxis] * coeffs_n
            R_mn = np.linalg.norm(R_m[:, np.newaxis] - R_n, axis=2)

            R_m_com = R_m - np.array([0, 0, centre_of_mass])
            R_n_com = R_n - np.array([0, 0, centre_of_mass])

            alpha_m_R_m_com = np.einsum("i, ij->ij", alphas_m, R_m_com)
            alpha_n_R_n_com = np.einsum("i, ij->ij", alphas_n, R_n_com)
                    
            OM_mn = coeffproduct_mn * (4 * product_mn / sum_mn ** 2) ** (3 / 4) * np.exp(-(product_mn / sum_mn) * R_mn ** 2)

            Rk_dipole = np.einsum("ijk,ij->ij", alpha_m_R_m_com[:, np.newaxis] + alpha_n_R_n_com, 1 / sum_mn)

            D[i,j] += np.einsum("mn,mn->", OM_mn, Rk_dipole)

    return D