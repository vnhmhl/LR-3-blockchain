#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

static inline int owner_of_row(int row, const std::vector<int>& displs, const std::vector<int>& counts) {
    // displs[r] = стартовая глобальная строка у ранга r
    // counts[r] = сколько строк у ранга r
    // найти r такое, что displs[r] <= row < displs[r] + counts[r]
    int p = (int)counts.size();
    // линейно (достаточно для лаб), можно бинпоиском
    for (int r = 0; r < p; ++r) {
        if (row >= displs[r] && row < displs[r] + counts[r]) return r;
    }
    return -1;
}

static inline int local_index(int global_row, int rank, const std::vector<int>& displs) {
    return global_row - displs[rank];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- параметры ---
    int n = 6; // по умолчанию
    if (argc >= 2) n = std::max(2, std::atoi(argv[1]));

    const int m = n + 1; // расширенная матрица A|b: n x (n+1)

    // --- распределение строк между процессами (block distribution) ---
    std::vector<int> counts(size, 0), displs(size, 0);
    int base = n / size, rem = n % size;
    for (int r = 0; r < size; ++r) counts[r] = base + (r < rem ? 1 : 0);
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r - 1] + counts[r - 1];

    int local_rows = counts[rank];
    std::vector<double> local_aug((size_t)local_rows * m, 0.0);

    // --- подготовка матрицы на root и Scatterv ---
    std::vector<double> global_aug; // только у rank 0
    if (rank == 0) {
        global_aug.resize((size_t)n * m);

        // Генерация диагонально доминирующей матрицы (чтобы обычно не было вырожденности)
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-5.0, 5.0);

        // A и b
        for (int i = 0; i < n; ++i) {
            double rowsum = 0.0;
            for (int j = 0; j < n; ++j) {
                double val = dist(rng);
                global_aug[(size_t)i * m + j] = val;
                rowsum += std::abs(val);
            }
            // усилить диагональ
            global_aug[(size_t)i * m + i] += rowsum + 10.0;

            // b
            global_aug[(size_t)i * m + n] = dist(rng);
        }
    }

    // Scatterv по строкам (каждая строка = m doubles)
    std::vector<int> sendcounts(size), senddispls(size);
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = counts[r] * m;
        senddispls[r] = displs[r] * m;
    }

    MPI_Scatterv(
        rank == 0 ? global_aug.data() : nullptr,
        sendcounts.data(),
        senddispls.data(),
        MPI_DOUBLE,
        local_aug.data(),
        local_rows * m,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    auto get_local_row_ptr = [&](int local_i) -> double* {
        return &local_aug[(size_t)local_i * m];
        };

    std::vector<double> pivot_row(m, 0.0);

    // --- прямой ход ---
    for (int k = 0; k < n; ++k) {
        // 1) поиск pivot (max |a[i][k]| для i>=k)
        double local_max = -1.0;
        int local_pivot_row = -1; // global index

        for (int li = 0; li < local_rows; ++li) {
            int gi = displs[rank] + li;
            if (gi < k) continue;
            double* row = get_local_row_ptr(li);
            double v = std::abs(row[k]);
            if (v > local_max) {
                local_max = v;
                local_pivot_row = gi;
            }
        }

        // MPI_MAXLOC для пары (value, index)
        struct { double val; int idx; } in, out;
        in.val = local_max;
        in.idx = (local_pivot_row == -1 ? -1 : local_pivot_row);

        // Если у процесса нет кандидатов, оставим val=-1, idx=-1
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        int pivot_global = out.idx;
        if (pivot_global < 0) {
            if (rank == 0) std::cerr << "Не удалось найти pivot на шаге k=" << k << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 2) swap строк k и pivot_global (если нужно)
        if (pivot_global != k) {
            int owner_k = owner_of_row(k, displs, counts);
            int owner_p = owner_of_row(pivot_global, displs, counts);

            int lk = (owner_k == rank) ? local_index(k, rank, displs) : -1;
            int lp = (owner_p == rank) ? local_index(pivot_global, rank, displs) : -1;

            if (owner_k == owner_p) {
                // обе строки у одного процесса
                if (rank == owner_k) {
                    double* row_k = get_local_row_ptr(lk);
                    double* row_p = get_local_row_ptr(lp);
                    for (int j = 0; j < m; ++j) std::swap(row_k[j], row_p[j]);
                }
            }
            else {
                // строки на разных процессах: обмен через Sendrecv
                if (rank == owner_k) {
                    std::vector<double> tmp(m);
                    double* row_k = get_local_row_ptr(lk);
                    std::copy(row_k, row_k + m, tmp.data());

                    MPI_Sendrecv(
                        tmp.data(), m, MPI_DOUBLE, owner_p, 1000 + k,
                        row_k, m, MPI_DOUBLE, owner_p, 2000 + k,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE
                    );
                }
                else if (rank == owner_p) {
                    std::vector<double> tmp(m);
                    double* row_p = get_local_row_ptr(lp);
                    std::copy(row_p, row_p + m, tmp.data());

                    MPI_Sendrecv(
                        tmp.data(), m, MPI_DOUBLE, owner_k, 2000 + k,
                        row_p, m, MPI_DOUBLE, owner_k, 1000 + k,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE
                    );
                }
            }
        }

        // 3) broadcast pivot-строки (теперь pivot строка находится в глобальной строке k)
        int owner_k = owner_of_row(k, displs, counts);
        if (rank == owner_k) {
            int lk = local_index(k, rank, displs);
            double* row_k = get_local_row_ptr(lk);
            std::copy(row_k, row_k + m, pivot_row.begin());
        }
        MPI_Bcast(pivot_row.data(), m, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);

        double pivot = pivot_row[k];
        if (std::abs(pivot) < 1e-15) {
            if (rank == 0) std::cerr << "Матрица вырожденная/почти вырожденная на шаге k=" << k << "\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        // 4) исключение (зануление ниже диагонали)
        for (int li = 0; li < local_rows; ++li) {
            int gi = displs[rank] + li;
            if (gi <= k) continue;

            double* row = get_local_row_ptr(li);
            double factor = row[k] / pivot;
            // row[j] -= factor * pivot_row[j], j = k..n
            row[k] = 0.0;
            for (int j = k + 1; j < m; ++j) {
                row[j] -= factor * pivot_row[j];
            }
        }
    }

    // --- собрать результат прямого хода на root ---
    if (rank == 0) {
        global_aug.assign((size_t)n * m, 0.0);
    }

    MPI_Gatherv(
        local_aug.data(), local_rows * m, MPI_DOUBLE,
        rank == 0 ? global_aug.data() : nullptr,
        sendcounts.data(), senddispls.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // --- обратный ход (на root) ---
    std::vector<double> x(n, 0.0);
    if (rank == 0) {
        for (int i = n - 1; i >= 0; --i) {
            double sum = global_aug[(size_t)i * m + n]; // b
            for (int j = i + 1; j < n; ++j) {
                sum -= global_aug[(size_t)i * m + j] * x[j];
            }
            double diag = global_aug[(size_t)i * m + i];
            if (std::abs(diag) < 1e-15) {
                std::cerr << "Нулевой диагональный элемент на i=" << i << "\n";
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            x[i] = sum / diag;
        }
    }

    // разослать x всем
    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- вывести решение на root ---
    if (rank == 0) {
        std::cout << "n=" << n << ", MPI processes=" << size << "\n";
        std::cout << "Solution x:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "x[" << i << "] = " << std::setprecision(10) << x[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
