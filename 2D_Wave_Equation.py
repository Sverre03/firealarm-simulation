import numpy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def N(xi, n):
    return (1 / 4 * np.array([1 - n - xi + n * xi,
                              1 - n + xi - n * xi,
                              1 + n + xi + n * xi,
                              1 + n - xi - n * xi])
            )

def J(x_vector, y_vector, xi, n):
    return 0.25 * np.array([
        [np.dot(np.array([-1 + n, 1 - n, 1 + n, -1 - n]), x_vector),
         np.dot(np.array([-1 + n, 1 - n, 1 + n, -1 - n]), y_vector)],

        [np.dot(np.array([-1 + xi, -1 - xi, 1 + xi, 1 - xi]), x_vector),
         np.dot(np.array([-1 + xi, -1 - xi, 1 + xi, 1 - xi]), y_vector)]
    ])

def J_inv(x_vector, y_vector, xi, n):
    return np.linalg.inv(J(x_vector, y_vector, xi, n))

def B(xi, n, J_inv):
    return 0.25 * J_inv @ np.array([
        [-1 + n, 1 - n, 1 + n, -1 - n],
        [-1 + xi, -1 - xi, 1 + xi, 1 - xi]
    ])

def K_element(x_vector, y_vector):
    p = 1 / np.sqrt(3)

    B_point_1 = B(-p, -p, J_inv(x_vector, y_vector, -p, -p))
    B_point_2 = B(p, -p, J_inv(x_vector, y_vector, p, -p))
    B_point_3 = B(p, p, J_inv(x_vector, y_vector, p, p))
    B_point_4 = B(-p, p, J_inv(x_vector, y_vector, -p, p))

    detJ1 = np.linalg.det(J(x_vector, y_vector, -p, -p))
    detJ2 = np.linalg.det(J(x_vector, y_vector, p, -p))
    detJ3 = np.linalg.det(J(x_vector, y_vector, p, p))
    detJ4 = np.linalg.det(J(x_vector, y_vector, -p, p))

    Element = ((

            detJ1 * B_point_1.T @ B_point_1 +
            detJ2 * B_point_2.T @ B_point_2 +
            detJ3 * B_point_3.T @ B_point_3 +
            detJ4 * B_point_4.T @ B_point_4)
    )
    return Element

# M_element without k**2 with Gauss quadrature
def detJ(point, x_vector, y_vector):
    return np.linalg.det(J(x_vector, y_vector, point[0], point[1]))

def M_element(x_vector, y_vector):
    weights = np.ones(4)

    p = 1 / np.sqrt(3)

    Gauss_points = np.array([[-p, -p], [p, -p], [p, p], [-p, p]])

    Element_matrix = np.zeros((4, 4))
    for i in range(4):
        Element_matrix += detJ(Gauss_points[i, :], x_vector, y_vector) * (
                weights[i] * np.outer(
            N(Gauss_points[i, :][0], Gauss_points[i, :][1]),
            N(Gauss_points[i, :][0], Gauss_points[i, :][1])))

    return Element_matrix

M_element(np.array([-1.0,1.0,1.0,-1.0]),np.array([-1.0,-1.0,1.0,1.0]))
K_element(np.array([-1.0,1.0,1.0,-1.0]),np.array([-1.0,-1.0,1.0,1.0]))

N_x = 100
M_x = 100
element_vector = np.arange(0, N_x * M_x, 1, dtype=int)

# element_num,local_nums, gloabal_nums
element_conectivity_table = np.zeros(((N_x * M_x), 5), dtype=int)
element_conectivity_table[:, 0] = element_vector

for i in range(np.shape(element_conectivity_table)[0]):
    mod_count = i // (N_x)
    element_conectivity_table[i, 1:5] = [i + mod_count, i + mod_count + 1, i + mod_count + N_x + 1 + 1,
                                         i + mod_count + 1 + N_x]

print(element_conectivity_table)

coordinates = np.zeros(((N_x + 1) * (M_x + 1), 4))
coordinates[:,3] = 1
coordinates[:,0] = np.arange(0, (N_x + 1) * (M_x + 1), 1)
#print(coordinats)
#x_values
coordinates[:,1] = (coordinates[:,0] % (M_x + 1))

#y_values
coordinates[:,2] = (coordinates[:,0] // (M_x + 1))
coordinates_dummy = coordinates.copy()

# reordering coordinats

# Fixed nodes given dette må fikses på bedre måte 1 fixed 0 free

coordinates[:, 3][0:(M_x + 1)] = np.zeros(N_x + 1)
coordinates[:, 3][(N_x + 1) * (M_x + 1) - (M_x + 1):] = np.zeros(N_x + 1)

for i in range(np.size(coordinates[:, 3])):
    #### AI
    row = i // (M_x + 1)
    col = i % (M_x + 1)
    if row == 0 or row == N_x or col == 0 or col == M_x:  # Boundary nodes
        coordinates[i, 3] = 0  # Fixed
    else:
        coordinates[i, 3] = 1  # Free
    ####

print(coordinates[:, 3], coordinates[:, 0])

ordering = np.vstack([coordinates[:, 0].T, coordinates[:, 3].T]).T

# print(ordering)
counter = 0
ordering_dummy = np.zeros((np.size(coordinates[:, 0]), 3))
ordering_dummy[:, 0] = np.arange(0, np.size(coordinates[:, 0]), 1)

for i in range(np.size(ordering[:, 0])):
    if ordering[i, 1] == 1:
        ordering_dummy[counter, 1:] = ordering[i, :]
        counter += 1

for i in range(np.size(ordering[:, 0])):
    if ordering[i, 1] == 0:
        ordering_dummy[counter, 1:] = ordering[i, :]
        counter += 1

ordering = ordering_dummy
print(ordering)


def satan_hest_kuk(i):
    a1, a2, a3, a4 = element_conectivity_table[i, 1], element_conectivity_table[i, 2], element_conectivity_table[i, 3], \
    element_conectivity_table[i, 4]
    return a1, a2, a3, a4


def Global_K_assembly(N, M, coordinats):
    Global_matrix_K = np.zeros(((np.shape(coordinats))[0], (np.shape(coordinats))[0]))
    for i in range(N * M):

        # print(np.shape(coordinats)[0])
        a = np.array(satan_hest_kuk(i))

        xy1, xy2, xy3, xy4 = coordinats[a[0], 1:3], coordinats[a[1], 1:3], coordinats[a[2], 1:3], coordinats[a[3], 1:3]

        K = K_element(np.array([xy1[0], xy2[0], xy3[0], xy4[0]]), np.array([xy1[1], xy2[1], xy3[1], xy4[1]]))

        if (i == 1):
            print(xy1, xy2, xy3, xy4)
            print(K)

        for k in range(4):
            for l in range(4):
                Global_matrix_K[a[k], a[l]] += K[k, l]
    return Global_matrix_K

# print(Global_K_assembly(N,M,coordinats))

def Global_M_assembly(N, M, coordinats):
    Global_matrix_M = np.zeros(((np.shape(coordinats))[0], (np.shape(coordinats))[0]))
    for i in range(N * M):
        # print(np.shape(coordinats)[0])
        a = np.array(satan_hest_kuk(i))

        xy1, xy2, xy3, xy4 = coordinats[a[0], 1:3], coordinats[a[1], 1:3], coordinats[a[2], 1:3], coordinats[a[3], 1:3]

        K = M_element(np.array([xy1[0], xy2[0], xy3[0], xy4[0]]), np.array([xy1[1], xy2[1], xy3[1], xy4[1]]))

        for k in range(4):
            for l in range(4):
                Global_matrix_M[a[k], a[l]] += K[k, l]

    return Global_matrix_M

Global_K, Global_M = Global_K_assembly(N_x, M_x, coordinates), Global_M_assembly(N_x, M_x, coordinates)

# assembling A matrix and vectors and forces
k = 0.1
A = Global_K - k ** 2 * Global_M
Force_vector = np.zeros(np.shape(A)[0])  # empty for now

def solve_system(x, y, threshold=60):
    idx = y * M_x + x
    Force_vector[idx] = 100

    Force_vector_dummy = Force_vector.copy()
    Force_vector_sorted = Force_vector.copy()

    # reordering
    A_sorted = np.zeros_like(A)
    for i in range(int(np.shape(A)[0])):
        A_sorted[int((ordering[i, 1])), :] = A[i, :]
        # print((ordering[i,:]))
        Force_vector_sorted[int((ordering[i, 1]))] = Force_vector_dummy[i]

    splice_number = np.count_nonzero(ordering[:, 2])
    A_reduced = A_sorted[0:splice_number, 0:splice_number]
    Force_vector_reduced = Force_vector_sorted[0:splice_number]
    # print(Force_vector_reduced)

    #denne ruta er AI

    # Step 1: Get free and fixed node indices
    free_nodes  = ordering[ordering[:,2] == 1, 0].astype(int)
    fixed_nodes = ordering[ordering[:,2] == 0, 0].astype(int)

    # Step 2: New ordering: free nodes first, then fixed nodes
    new_order = np.hstack([free_nodes, fixed_nodes])

    # Step 3: Reorder A and Force vector
    A_sorted = A[np.ix_(new_order, new_order)]
    Force_vector_sorted = Force_vector[new_order]

    # Step 4: Reduced system for free nodes only
    n_free = len(free_nodes)
    A_reduced = A_sorted[:n_free, :n_free]
    Force_vector_reduced = Force_vector_sorted[:n_free]

    # Step 5: Solve system
    A_sparse = sp.csr_matrix(A_reduced)
    M_inv = sp.diags(1.0 / A_sparse.diagonal())

    solution_free, info = spla.cg(A_sparse, Force_vector_reduced, M=M_inv, rtol=1e-8)

    result = numpy.asarray((abs(solution_free) > threshold).astype(int))

    return result.mean()

result = solve_system(0, 10, 60)
