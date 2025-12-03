import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ===============================
# LOADING
# ===============================

def load_image_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            number_strings = line.strip().split() 
            if not number_strings:
                continue
            row = [float(s) for s in number_strings] 
            matrix.append(row)
    return matrix

def crop_matrix_simple(matrix, N):
    """
    Ritaglia sempre la finestra NxN a partire dall'angolo in alto a sinistra.
    Assumiamo N <= dimensioni della matrice.
    """
    cropped = [row[:N] for row in matrix[:N]]
    return cropped
# ===============================
# NAIVE BINARIZATION METHODS
# ===============================

def sauvola_naive(in_image, w, k=0.06, R=128):
    rows = len(in_image)
    columns = len(in_image[0])
    bin_image = [[0] * columns for _ in range(rows)]
    d = w // 2
    for x in range(rows):
        for y in range(columns):
            x1 = max(0, x-d)
            x2 = min(rows, x+d+1)
            y1 = max(0, y-d)
            y2 = min(columns, y+d+1)
            pixels = [in_image[i][j]
                      for i in range(x1, x2)
                      for j in range(y1, y2)]
            n = len(pixels)
            mean = sum(pixels) / n
            std = (sum((p - mean) ** 2 for p in pixels) / n) ** 0.5
            threshold = mean * (1 + k*(std/R - 1))
            bin_image[x][y] = in_image[x][y] > threshold
    return bin_image


def niblack_naive(in_image, w, k=-2):
    rows = len(in_image)
    columns = len(in_image[0])
    bin_image = [[0] * columns for _ in range(rows)]
    d = w // 2
    for x in range(rows):
        for y in range(columns):
            x1 = max(0, x-d)
            x2 = min(rows, x+d+1)
            y1 = max(0, y-d)
            y2 = min(columns, y+d+1)
            pixels = [in_image[i][j]
                      for i in range(x1, x2)
                      for j in range(y1, y2)]
            n = len(pixels)
            mean = sum(pixels) / n
            std = (sum((p - mean) ** 2 for p in pixels) / n) ** 0.5
            threshold = mean + k*std 
            bin_image[x][y] = in_image[x][y] > threshold
    return bin_image


def bernsen_naive(in_image, w, contrast_threshold=15):
    rows = len(in_image)
    columns = len(in_image[0])
    bin_image = [[0] * columns for _ in range(rows)]
    d = w // 2
    for x in range(rows):
        for y in range(columns):
            x1 = max(0, x - d)
            x2 = min(rows, x + d + 1)
            y1 = max(0, y - d)
            y2 = min(columns, y + d + 1)
            pixels = [in_image[i][j]
                      for i in range(x1, x2)
                      for j in range(y1, y2)]
            Imax = max(pixels)
            Imin = min(pixels)
            C = Imax - Imin 
            if C < contrast_threshold:
                T = sum(pixels) / len(pixels)
            else:
                T = (Imax + Imin) / 2 
            bin_image[x][y] = 1 if in_image[x][y] > T else 0
    return bin_image

# ===============================
# FAST METHODS
# ===============================

def proposed_int_sum(in_image, w, k=0.3):
    rows = len(in_image)
    cols = len(in_image[0])
    bin_image = [[0] * cols for _ in range(rows)]
    integral = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            val = in_image[i-1][j-1]
            integral[i][j] = val + integral[i-1][j] + integral[i][j-1] - integral[i-1][j-1]
    d = w // 2
    # eps = 1e-9
    for x in range(rows):
        for y in range(cols):
            x1 = max(0, x - d)
            x2 = min(rows - 1, x + d)
            y1 = max(0, y - d)
            y2 = min(cols - 1, y + d)
            local_sum = (integral[x2 + 1][y2 + 1]
                         - integral[x1][y2 + 1]
                         - integral[x2 + 1][y1]
                         + integral[x1][y1])
            window_size = (x2 - x1 + 1) * (y2 - y1 + 1)
            local_mean = local_sum / window_size
            delta = in_image[x][y] - local_mean
            threshold = 0
            if (delta == 1):
                threshold = local_mean * (1 + k * (delta / (2 - delta)))
            else:
                threshold = local_mean * (1 + k * (delta / (1 - delta)))

            # threshold = local_mean * (1 + k * (delta / (1 - abs(delta) + eps)))

            bin_image[x][y] = 1 if in_image[x][y] > threshold else 0
    return bin_image


def global_fast(in_image, w=None):
    rows = len(in_image)
    cols = len(in_image[0])
    old_threshold = sum(sum(row) for row in in_image) / (rows * cols) 
    max_iters = 100
    for iteration in range(max_iters):
        higher = [] 
        lower = []   
        for i in range(rows):
            row = in_image[i]
            for j in range(cols):
                v = row[j]
                if v >= old_threshold:
                    higher.append(v)
                else:
                    lower.append(v)
        if len(higher) == 0:
            higher_mean = 0.0
        else:
            higher_mean = sum(higher) / len(higher)
        if len(lower) == 0:
            lower_mean = 0.0
        else:
            lower_mean = sum(lower) / len(lower)
        new_threshold = 0.5 * (higher_mean + lower_mean)
        if abs(new_threshold - old_threshold) <= 0.1 :
            break
        old_threshold = new_threshold
    else:
        pass
    bin_image = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            bin_image[i][j] = 1 if in_image[i][j] > new_threshold else 0
    return bin_image

# ===============================
# BENCHMARKING + PLOT
# ===============================

def save_binary_matrix(matrix, filename):
    with open(filename, "w") as f:
        for row in matrix:
            f.write(" ".join(str(int(v) * 255) for v in row) + "\n")


def benchmark_binarization(path):
    in_image = load_image_matrix(file_path=path)
    crop_sizes = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    window_sizes = [3, 7, 11, 15, 19, 23, 27, 31, 35]
    num_runs = 10

    naive_methods = {
        'Sauvola': sauvola_naive,
        'Niblack': niblack_naive,
        'Bernsen': bernsen_naive,
    }

    fast_methods = {
        'Proposed': proposed_int_sum,
        'Global': global_fast
    }

    naive_times = np.zeros((len(naive_methods), len(window_sizes), len(crop_sizes)))
    fast_times = np.zeros((len(fast_methods),  len(window_sizes), len(crop_sizes)))

    print("Benchmarking methods...")

    
    for ci, crop_size in enumerate(crop_sizes):
        cropped_image=crop_matrix_simple(in_image, crop_size)
        for wi, w in enumerate(window_sizes):
            print(f"\n== Testing w = {w} == n = {crop_size}")
            for mi, (name, method) in enumerate(naive_methods.items()):
                t = []
                for r in range(num_runs):
                    start = time.perf_counter()
                    bin_image = method(cropped_image, w)
                    t.append(time.perf_counter() - start)
                naive_times[mi, wi, ci] = np.mean(t)
                print(f"{name}: {naive_times[mi, wi, ci]:.4f} s")
            #     save_binary_matrix(bin_image, f"{name.replace(' ', '_')}_w{w}.txt")

            for mi, (name, method) in enumerate(fast_methods.items()):
                t = []
                for r in range(num_runs):
                    start = time.perf_counter()
                    bin_image = method(cropped_image, w)
                    t.append(time.perf_counter() - start)
                fast_times[mi, wi, ci] = np.mean(t)
                print(f"{name}: {fast_times[mi, wi, ci]:.4f} s")
                #save_binary_matrix(bin_image, f"{name.replace(' ', '_')}_w{w}.txt")

   
    num_naive_methods = len(naive_methods)
    num_fast_methods = len(fast_methods)
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan']  # più colori se servono

    # Meshgrid
    W, N = np.meshgrid(window_sizes, crop_sizes, indexing='ij')

    fig = plt.figure(figsize=(16, 6))

    # Grafico 3D principale
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # Plot dei metodi naive
    for mi, name in enumerate(naive_methods.keys()):
        T = naive_times[mi, :, :]
        ax.plot_surface(W, N, T, color=colors[mi % len(colors)], alpha=0.6, label=f"{name}")

    # Plot dei metodi fast
    for mi, name in enumerate(fast_methods.keys()):
        T = fast_times[mi, :, :]
        ax.plot_surface(W, N, T, color=colors[(mi + num_naive_methods) % len(colors)], alpha=0.6, label=f"{name}")

    ax.set_xlabel('Window size w')
    ax.set_ylabel('Crop size N')
    ax.set_zlabel('Runtime (s)')
    ax.set_title('Runtime surface per method')

    # Slice laterali: avg vs w
    ax2 = fig.add_subplot(2, 2, 2)
    for mi, name in enumerate(naive_methods.keys()):
        T = naive_times[mi, :, :]
        ax2.plot(window_sizes, T.mean(axis=1), marker='o', color=colors[mi % len(colors)], label=f"{name}")
    for mi, name in enumerate(fast_methods.keys()):
        T = fast_times[mi, :, :]
        ax2.plot(window_sizes, T.mean(axis=1), marker='o', color=colors[(mi + num_naive_methods) % len(colors)], label=f"{name}")
    ax2.set_xlabel('w')
    ax2.set_ylabel('Runtime (s)')
    ax2.set_title('Average t vs w')
    ax2.legend()

    # Slice laterali: avg vs N
    ax3 = fig.add_subplot(2, 2, 4)
    for mi, name in enumerate(naive_methods.keys()):
        T = naive_times[mi, :, :]
        ax3.plot(crop_sizes, T.mean(axis=0), marker='o', color=colors[mi % len(colors)], label=f"{name}")
    for mi, name in enumerate(fast_methods.keys()):
        T = fast_times[mi, :, :]
        ax3.plot(crop_sizes, T.mean(axis=0), marker='o', color=colors[(mi + num_naive_methods) % len(colors)], label=f"{name}")
    ax3.set_xlabel('N')
    ax3.set_ylabel('Runtime (s)')
    ax3.set_title('Average t vs N')
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    benchmark_binarization(r"NileBend.txt")
