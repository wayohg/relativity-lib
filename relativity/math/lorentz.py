import numpy as np
from relativity.utils import smart_array, smart_sqrt  # FIX: absolute import


def gamma(v, c):
    v = smart_array(v, dtype=float)
    beta2 = np.dot(v, v) / c**2
    if beta2 >= 1:
        raise ValueError("Velocidad superlumínica no permitida.")
    return 1.0 / smart_sqrt(1 - beta2)


def boost_matrix(v, c):
    v = smart_array(v, dtype=float)
    v2 = np.dot(v, v)

    if v2 == 0:
        return np.eye(4)

    g = gamma(v, c)
    beta = v / c

    Lambda = np.eye(4)

    Lambda[0, 0] = g
    Lambda[0, 1:] = -g * beta
    Lambda[1:, 0] = -g * beta

    beta2 = np.dot(beta, beta)
    outer = np.outer(beta, beta)
    Lambda[1:, 1:] += (g - 1) * outer / beta2  # FIX: was dividing by np.dot(beta,beta) correctly but only when v2!=0 — safe here

    return Lambda


def relativistic_velocity_addition(u, v, c):
    """
    Suma relativista general 3D.

    u: velocidad del objeto en S
    v: velocidad del marco S' respecto a S
    """
    u = smart_array(u, dtype=float)
    v = smart_array(v, dtype=float)

    v2 = np.dot(v, v)
    if v2 == 0:
        return u

    u_parallel = (np.dot(u, v) / v2) * v
    u_perp = u - u_parallel

    gamma_v = 1.0 / smart_sqrt(1 - v2 / c**2)
    uv_dot = np.dot(u, v)

    result_parallel = (u_parallel + v) / (1 + uv_dot / c**2)
    result_perp = u_perp / (gamma_v * (1 + uv_dot / c**2))

    return result_parallel + result_perp


def simultaneity_velocity(delta_t, delta_r, c):
    """
    Velocidad en 1D para que dos eventos sean simultáneos en el nuevo marco.
    """
    if delta_r == 0:
        raise ValueError("Eventos coincidentes espacialmente.")

    v = c**2 * delta_t / delta_r

    if abs(v) >= c:
        raise ValueError("Requiere velocidad superlumínica.")

    return v


def time_dilation(proper_time, v, c):
    v = smart_array(v, dtype=float)
    v2 = np.dot(v, v)
    gamma_v = 1.0 / smart_sqrt(1 - v2 / c**2)
    return gamma_v * proper_time


def length_contraction(length_vector, v, c):
    """
    Contracción relativista general 3D.
    length_vector: longitud propia (3-vector)
    v: velocidad relativa (3-vector)
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
