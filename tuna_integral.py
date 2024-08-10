import numpy as np
import scipy.special as special

def special_function(x): return np.where(x == 0, 1, (0.5 * (np.pi / (x+0.00000000000000001)) ** 0.5) * special.erf((x+0.00000000000000001) ** 0.5))

   
def evaluate_integrals(orbitals, Z_list, atomic_coords):

    nbasis = len(orbitals)

    V_EE = np.zeros([nbasis, nbasis, nbasis, nbasis], dtype=float)
    S = np.zeros([nbasis, nbasis])
    T = np.zeros([nbasis, nbasis])
    V_NE = np.zeros([nbasis, nbasis])
    

    for i in range(nbasis):
        for j in range(i, nbasis): 
            
            for ii in range(len(orbitals[i])):
                for jj in range(len(orbitals[j])):

                    coeff_prod_ij = orbitals[i][ii].coeff * orbitals[j][jj].coeff
                    s_ij = orbitals[i][ii].alpha + orbitals[j][jj].alpha
                    p_ij = orbitals[i][ii].alpha * orbitals[j][jj].alpha
                    R_ij = np.linalg.norm(orbitals[i][ii].coordinates - orbitals[j][jj].coordinates)
                                    
                    OM_ij = coeff_prod_ij * (4 * p_ij / s_ij ** 2) ** (3 / 4) * np.exp(-(p_ij / s_ij) * np.dot(R_ij, R_ij))
                    Rk = (orbitals[i][ii].alpha * orbitals[i][ii].coordinates + orbitals[j][jj].alpha * orbitals[j][jj].coordinates) / s_ij

                    S[i,j] += OM_ij
                    T[i,j] += OM_ij * (p_ij / s_ij) * (3 - (2 * p_ij * R_ij**2) / s_ij)

                    for atom in range(len(Z_list)):

                        dfunc_to_atom_ij = np.linalg.norm(Rk - atomic_coords[atom]) ** 2
                        V_NE[i,j] += -Z_list[atom] * OM_ij * special_function(s_ij * dfunc_to_atom_ij) * 2 * np.sqrt(s_ij / np.pi)
            
            S[j,i] = S[i,j]
            T[j,i] = T[i,j]
            V_NE[j,i] = V_NE[i,j]

            for k in range(nbasis):
                for l in range(k, nbasis): 

                    V_EE_value = 0

                    for ii in range(len(orbitals[i])):
                        for jj in range(len(orbitals[j])):
                            for kk in range(len(orbitals[k])):
                                for ll in range(len(orbitals[l])):
                                
                                    coeff_prod_ij = orbitals[i][ii].coeff * orbitals[j][jj].coeff
                                    s_ij = orbitals[i][ii].alpha + orbitals[j][jj].alpha
                                    p_ij = orbitals[i][ii].alpha * orbitals[j][jj].alpha
                                    R_ij = np.linalg.norm(orbitals[i][ii].coordinates - orbitals[j][jj].coordinates)

                                    coeff_prod_kl = orbitals[k][kk].coeff * orbitals[l][ll].coeff
                                    s_kl = orbitals[k][kk].alpha + orbitals[l][ll].alpha
                                    p_kl = orbitals[k][kk].alpha * orbitals[l][ll].alpha
                                    R_kl = np.linalg.norm(orbitals[k][kk].coordinates - orbitals[l][ll].coordinates)

                                    OM_ij = coeff_prod_ij * (4 * p_ij / s_ij ** 2) ** (3 / 4) * np.exp(-(p_ij / s_ij) * np.dot(R_ij, R_ij))
                                    OM_kl = coeff_prod_kl * (4 * p_kl / s_kl ** 2) ** (3 / 4) * np.exp(-(p_kl / s_kl) * np.dot(R_kl, R_kl)) 

                                    prod_over_sum = s_ij * s_kl / (s_ij + s_kl)
                                    Rk = (orbitals[i][ii].alpha * orbitals[i][ii].coordinates + orbitals[j][jj].alpha * orbitals[j][jj].coordinates) / s_ij
                                    
                                    Rl = (orbitals[k][kk].alpha * orbitals[k][kk].coordinates + orbitals[l][ll].alpha * orbitals[l][ll].coordinates) / s_kl

                                    V_EE_value += 2 / np.sqrt(np.pi) * np.sqrt(prod_over_sum) * special_function(prod_over_sum * np.dot((Rk - Rl),(Rk - Rl))) * OM_ij * OM_kl
                
                
                    V_EE[i, j, k, l] = V_EE_value
                    V_EE[j, i, l, k] = V_EE_value  
                    V_EE[j, i, k, l] = V_EE_value 
                    V_EE[i, j, l, k] = V_EE_value

    return S, T, V_NE, V_EE


def evaluate_dipole_integrals(orbitals, centre_of_mass):

    nbasis = len(orbitals)

    D = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(i, nbasis): 
            
            for ii in range(len(orbitals[i])):
                for jj in range(len(orbitals[j])):

                    coeff_prod_ij = orbitals[i][ii].coeff * orbitals[j][jj].coeff
                    s_ij = orbitals[i][ii].alpha + orbitals[j][jj].alpha
                    p_ij = orbitals[i][ii].alpha * orbitals[j][jj].alpha
                    R_ij = np.linalg.norm(orbitals[i][ii].coordinates - orbitals[j][jj].coordinates)
                                    
                    OM_ij = coeff_prod_ij * (4 * p_ij / s_ij ** 2) ** (3 / 4) * np.exp(-(p_ij / s_ij) * np.dot(R_ij, R_ij))
                    Rk_dipole = (orbitals[i][ii].alpha * (orbitals[i][ii].coordinates[2] - centre_of_mass) + orbitals[j][jj].alpha * (orbitals[j][jj].coordinates[2] - centre_of_mass)) / s_ij

                    D[i,j] += OM_ij * Rk_dipole

            D[j,i] = D[i,j]

    return D



def calculate_integrals(orbitals, Z_list, atomic_coords):

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

            alpha_m_R_m = np.einsum("i, ij->ij", alphas_m, R_m)
            alpha_n_R_n = np.einsum("i, ij->ij", alphas_n, R_n)

            Rk = np.einsum("ijk,ij->ij",(alpha_m_R_m[:, np.newaxis] + alpha_n_R_n), 1 / sum_mn)
                    
            OM_mn = coeffproduct_mn * (4 * product_mn / sum_mn ** 2) ** (3 / 4) * np.exp(-(product_mn / sum_mn) * R_mn ** 2)
            
            S[i,j] = np.einsum("mn->", OM_mn)
            T[i,j] = np.einsum("mn,mn,mn->", OM_mn, (product_mn / sum_mn), (3 - (2 * product_mn * R_mn**2) / sum_mn))

            for atom in range(len(Z_list)):

                dfunc_to_atom_mn = (Rk - atomic_coords[atom][2]) ** 2

                V_NE[i,j] += -Z_list[atom] * np.einsum("mn,mn,mn->",OM_mn,special_function(sum_mn * dfunc_to_atom_mn), 2 * np.sqrt(sum_mn / np.pi))
            
            S[j,i] = S[i,j]
            T[j,i] = T[i,j]
            V_NE[j,i] = V_NE[i,j] 
            
            for k in range(nbasis):
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


    return S, T, V_NE, V_EE



def calculate_one_electron_integrals(orbitals, Z_list, atomic_coords):

    nbasis = len(orbitals)

    S = np.zeros([nbasis, nbasis])
    T = np.zeros([nbasis, nbasis])
    V_NE = np.zeros([nbasis, nbasis])

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

            alpha_m_R_m = np.einsum("i, ij->ij", alphas_m, R_m)
            alpha_n_R_n = np.einsum("i, ij->ij", alphas_n, R_n)

            Rk = np.einsum("ijk,ij->ij",(alpha_m_R_m[:, np.newaxis] + alpha_n_R_n), 1 / sum_mn)
                    
            OM_mn = coeffproduct_mn * (4 * product_mn / sum_mn ** 2) ** (3 / 4) * np.exp(-(product_mn / sum_mn) * R_mn ** 2)
            
            S[i,j] = np.einsum("mn->", OM_mn)
            T[i,j] = np.einsum("mn,mn,mn->", OM_mn, (product_mn / sum_mn), (3 - (2 * product_mn * R_mn**2) / sum_mn))

            for atom in range(len(Z_list)):

                dfunc_to_atom_mn = (Rk - atomic_coords[atom][2]) ** 2

                V_NE[i,j] += -Z_list[atom] * np.einsum("mn,mn,mn->",OM_mn,special_function(sum_mn * dfunc_to_atom_mn), 2 * np.sqrt(sum_mn / np.pi))
            
            S[j,i] = S[i,j]
            T[j,i] = T[i,j]
            V_NE[j,i] = V_NE[i,j] 

    return S, T, V_NE