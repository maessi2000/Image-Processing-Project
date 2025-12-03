#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

/* ============================================================
   UTILS
   ============================================================
 */

vector<vector<double>> load_matrix(const string& path) {
  ifstream f(path);
  vector<vector<double>> M;
  string line;
  size_t max_cols = 0;

  while (getline(f, line)) {
    stringstream ss(line);
    vector<double> row;
    double x;
    while (ss >> x) row.push_back(x);
    if (!row.empty()) {
      max_cols = max(max_cols, row.size());
      M.push_back(row);
    }
  }

  // normalizza larghezza righe
  for (auto& r : M) r.resize(max_cols, 0.0);

  return M;
}

vector<vector<double>> crop_matrix(
    const vector<vector<double>>& M, int N) {
  int R = M.size();
  int C = R ? M[0].size() : 0;

  int RR = min(R, N);
  int CC = min(C, N);

  vector<vector<double>> CROP(RR, vector<double>(CC));

  for (int i = 0; i < RR; i++)
    for (int j = 0; j < CC; j++) CROP[i][j] = M[i][j];

  return CROP;
}

/* ============================================================
   INTEGRAL IMAGE
   ============================================================
 */

vector<vector<double>> integral_image(
    const vector<vector<double>>& img) {
  int R = img.size();
  int C = img[0].size();

  vector<vector<double>> I(R + 1, vector<double>(C + 1, 0.0));

  for (int i = 1; i <= R; i++) {
    double row_sum = 0.0;
    for (int j = 1; j <= C; j++) {
      row_sum += img[i - 1][j - 1];
      I[i][j] = I[i - 1][j] + row_sum;
    }
  }
  return I;
}

inline double window_sum(const vector<vector<double>>& I,
                         int x1, int y1, int x2, int y2) {
  return I[x2 + 1][y2 + 1] - I[x1][y2 + 1] - I[x2 + 1][y1] +
         I[x1][y1];
}

/* ============================================================
   FAST METHODS (PARALLELIZZATI)
   ============================================================
 */

vector<vector<int>> tanh_fast(
    const vector<vector<double>>& img, int w, double k = 0.3,
    double a = 1.0) {
  int R = img.size();
  int C = img[0].size();
  vector<vector<int>> out(R, vector<int>(C));

  auto I = integral_image(img);
  int d = w / 2;

#pragma omp parallel for schedule(static)
  for (int x = 0; x < R; x++) {
    for (int y = 0; y < C; y++) {
      int x1 = max(0, x - d), y1 = max(0, y - d);
      int x2 = min(R - 1, x + d), y2 = min(C - 1, y + d);

      double S = window_sum(I, x1, y1, x2, y2);
      int n = (x2 - x1 + 1) * (y2 - y1 + 1);

      double mean = S / n;
      double delta = img[x][y] - mean;

      double thr = mean * (1.0 + k * tanh(a * delta));

      out[x][y] = img[x][y] > thr ? 1 : 0;
    }
  }
  return out;
}

vector<vector<int>> proposed_fast(
    const vector<vector<double>>& img, int w, double k = 0.3) {
  int R = img.size();
  int C = img[0].size();
  vector<vector<int>> out(R, vector<int>(C));

  auto I = integral_image(img);
  int d = w / 2;

#pragma omp parallel for schedule(static)
  for (int x = 0; x < R; x++) {
    for (int y = 0; y < C; y++) {
      int x1 = max(0, x - d), y1 = max(0, y - d);
      int x2 = min(R - 1, x + d), y2 = min(C - 1, y + d);

      double S = window_sum(I, x1, y1, x2, y2);
      int n = (x2 - x1 + 1) * (y2 - y1 + 1);

      double mean = S / n;
      double delta = img[x][y] - mean;

      double thr =
          (fabs(delta - 1.0) < 1e-12)
              ? mean * (1 + k * (delta / (2 - delta)))
              : mean * (1 + k * (delta / (1 - delta)));

      out[x][y] = img[x][y] > thr ? 1 : 0;
    }
  }
  return out;
}

vector<vector<int>> otsu_global(
    const vector<vector<double>>& img) {
  int R = img.size();
  int C = img[0].size();

  vector<vector<int>> out(R, vector<int>(C));

  // Istogramma dei livelli [0..255]
  array<long long, 256> hist{};
  hist.fill(0);

// Costruisci histogramma (parallelizzato)
#pragma omp parallel
  {
    array<long long, 256> local_hist{};
    local_hist.fill(0);

#pragma omp for nowait
    for (int i = 0; i < R; i++)
      for (int j = 0; j < C; j++) {
        int v = (int)round(img[i][j]);
        v = max(0, min(255, v));
        local_hist[v]++;
      }

// riduzione
#pragma omp critical
    {
      for (int i = 0; i < 256; i++) hist[i] += local_hist[i];
    }
  }

  long long total = (long long)R * C;
  double sum_total = 0.0;

  for (int i = 0; i < 256; i++) sum_total += i * hist[i];

  long long wB = 0;
  long long wF = 0;

  double sumB = 0.0;
  double max_var = -1.0;
  int best_thr = 0;

  for (int t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB == 0) continue;

    wF = total - wB;
    if (wF == 0) break;

    sumB += t * hist[t];

    double mB = sumB / wB;
    double mF = (sum_total - sumB) / wF;

    double between =
        (double)wB * (double)wF * (mB - mF) * (mB - mF);

    if (between > max_var) {
      max_var = between;
      best_thr = t;
    }
  }

// Applica threshold globale
#pragma omp parallel for
  for (int i = 0; i < R; i++)
    for (int j = 0; j < C; j++)
      out[i][j] = img[i][j] > best_thr ? 1 : 0;

  return out;
}

/* ============================================================
   NAIVE METHODS (PARALLELIZZATI)
   ============================================================
 */

vector<vector<int>> sauvola_naive(
    const vector<vector<double>>& img, int w, double k = 0.06,
    double Rval = 128) {
  int R = img.size(), C = img[0].size();
  vector<vector<int>> out(R, vector<int>(C));

  int d = w / 2;

#pragma omp parallel for schedule(static)
  for (int x = 0; x < R; x++) {
    for (int y = 0; y < C; y++) {
      int x1 = max(0, x - d), y1 = max(0, y - d);
      int x2 = min(R - 1, x + d), y2 = min(C - 1, y + d);

      double sum = 0.0, sum2 = 0.0;
      int cnt = 0;

      for (int i = x1; i <= x2; i++)
        for (int j = y1; j <= y2; j++) {
          double v = img[i][j];
          sum += v;
          sum2 += v * v;
          cnt++;
        }

      double mean = sum / cnt;
      double var = sum2 / cnt - mean * mean;
      double std = var > 0 ? sqrt(var) : 0;

      double thr = mean * (1 + k * (std / Rval - 1));

      out[x][y] = img[x][y] > thr ? 1 : 0;
    }
  }

  return out;
}

vector<vector<int>> niblack_naive(
    const vector<vector<double>>& img, int w,
    double k = -0.2) {
  int R = img.size(), C = img[0].size();
  vector<vector<int>> out(R, vector<int>(C));

  int d = w / 2;

#pragma omp parallel for schedule(static)
  for (int x = 0; x < R; x++) {
    for (int y = 0; y < C; y++) {
      int x1 = max(0, x - d), y1 = max(0, y - d);
      int x2 = min(R - 1, x + d), y2 = min(C - 1, y + d);

      double sum = 0.0, sum2 = 0.0;
      int cnt = 0;

      for (int i = x1; i <= x2; i++)
        for (int j = y1; j <= y2; j++) {
          double v = img[i][j];
          sum += v;
          sum2 += v * v;
          cnt++;
        }

      double mean = sum / cnt;
      double var = sum2 / cnt - mean * mean;
      double std = sqrt(max(0.0, var));

      double thr = mean + k * std;

      out[x][y] = img[x][y] > thr ? 1 : 0;
    }
  }

  return out;
}

vector<vector<int>> bernsen_naive(
    const vector<vector<double>>& img, int w,
    double contrast_thr = 15) {
  int R = img.size(), C = img[0].size();
  vector<vector<int>> out(R, vector<int>(C));

  int d = w / 2;

#pragma omp parallel for schedule(static)
  for (int x = 0; x < R; x++) {
    for (int y = 0; y < C; y++) {
      int x1 = max(0, x - d), y1 = max(0, y - d);
      int x2 = min(R - 1, x + d), y2 = min(C - 1, y + d);

      double Imin = 1e300, Imax = -1e300;
      double sum = 0.0;
      int cnt = 0;

      for (int i = x1; i <= x2; i++)
        for (int j = y1; j <= y2; j++) {
          double v = img[i][j];
          Imin = min(Imin, v);
          Imax = max(Imax, v);
          sum += v;
          cnt++;
        }

      double Cnt = Imax - Imin;
      double thr = (Cnt < contrast_thr) ? (sum / cnt)
                                        : 0.5 * (Imax + Imin);

      out[x][y] = img[x][y] > thr ? 1 : 0;
    }
  }

  return out;
}

/* ============================================================
   BENCHMARK
   ============================================================
 */

double now() {
  using namespace chrono;
  return duration_cast<duration<double>>(
             steady_clock::now().time_since_epoch())
      .count();
}

int main() {
  string path = "input/NileBend.txt";
  auto img = load_matrix(path);

  vector<int> crop_sizes = {100,  250,  500,  1000, 1500,
                            2000, 2500, 3000, 3500, 4000};
  vector<int> window_sizes = {3,  7,  11, 15, 19,
                              23, 27, 31, 35};
  int num_runs = 10;

  cout << "Threads: " << omp_get_max_threads() << "\n";

  for (int N : crop_sizes) {
    auto C = crop_matrix(img, N);
    cout << "\nCrop = " << C.size() << " x " << C[0].size()
         << "\n";

    for (int w : window_sizes) {
      double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;

      for (int r = 0; r < num_runs; r++) {
        double a0 = now();
        tanh_fast(C, w);
        t1 += now() - a0;
        double b0 = now();
        proposed_fast(C, w);
        t2 += now() - b0;
        double c0 = now();
        sauvola_naive(C, w);
        t3 += now() - c0;
        double d0 = now();
        niblack_naive(C, w);
        t4 += now() - d0;
        double e0 = now();
        bernsen_naive(C, w);
        t5 += now() - e0;
        double f0 = now();
        otsu_global(C);
        t6 += now() - f0;
      }

      cout << "\nw=" << w << "\n";
      cout << "  Tanh-fast:     " << t1 / num_runs << " s\n";
      cout << "  Proposed-fast: " << t2 / num_runs << " s\n";
      cout << "  Otsu-fast:     " << t6 / num_runs << " s\n";
      cout << "  Sauvola:       " << t3 / num_runs << " s\n";
      cout << "  Niblack:       " << t4 / num_runs << " s\n";
      cout << "  Bernsen:       " << t5 / num_runs << " s\n";
    }
  }

  return 0;
}