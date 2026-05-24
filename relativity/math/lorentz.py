import numpy as np
from utils import smart_array, smart_sqrt

def gamma(v, c):
    beta2 = np.dot(v, v) / c**2
    if beta2 >= 1:
        raise ValueError("Velocidad superlumínica no permitida.")
    return 1.0 / smart_sqrt(1 - beta2)

def boost_matrix(v, c):
    v = smart_array(v)
    v2 = np.dot(v, v)

    if v2 == 0:
        return np.eye(4)

    g = gamma(v, c)
    beta = v / c

    Lambda = np.eye(4)

    # Parte temporal
    Lambda[0,0] = g
    Lambda[0,1:] = -g * beta
    Lambda[1:,0] = -g * beta

    # Parte espacial
    outer = np.outer(beta, beta)
    Lambda[1:,1:] += (g - 1) * outer / np.dot(beta, beta)

    return Lambda

def relativistic_velocity_addition(u, v, c):
    """
    Suma relativista general 3D.
    
    u: velocidad del objeto en S
    v: velocidad del marco S' respecto a S
    """
    u = smart_array(u, dtype=float)
    v = smart_array(v, dtype=float)

    u_parallel = (np.dot(u, v) / np.dot(v, v)) * v
    u_perp = u - u_parallel

    v2 = np.dot(v, v)
    if v2 == 0:
        return u

    gamma_v = 1.0 / smart_sqrt(1 - v2 / c**2)

    result_parallel = (u_parallel + v) / (1 + np.dot(u, v)/c**2)
    result_perp = u_perp / (gamma_v * (1 + np.dot(u, v)/c**2))

    return result_parallel + result_perp


def simultaneity_velocity(delta_t, delta_r, c):
    """
    Devuelve la velocidad en 1D para que dos eventos
    sean simultáneos en el nuevo marco.
    """
    if delta_r == 0:
        raise ValueError("Eventos coincidentes espacialmente.")

    v = c**2 * delta_t / delta_r

    if abs(v) >= c:
        raise ValueError("Requiere velocidad superlumínica.")

    return v

def time_dilation(proper_time, v, c):
    v2 = np.dot(v, v)
    gamma_v = 1.0 / smart_sqrt(1 - v2 / c**2)
    return gamma_v * proper_time

def length_contraction(length_vector, v, c):
    """
    Contracción relativista general 3D.
    length_vector: longitud propia
    v: velocidad relativa
    """
    L = smart_array(length_vector, dtype=float)
    v = smart_array(v, dtype=float)

    v2 = np.dot(v, v)
    if v2 == 0:
        return L

    gamma_v = 1.0 / smart_sqrt(1 - v2 / c**2)

    L_parallel = (np.dot(L, v) / v2) * v
    L_perp = L - L_parallel

    return L_perp + L_parallel / gamma_v