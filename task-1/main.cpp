#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <functional>
#include <omp.h>

using namespace std;

class approximator {
    const int bs, nb;

    const double h, eps, zero = 1e-15;

    vector<vector<double>> u;
    vector<vector<double>> f;

    const function<double(double, double)> fun;

    double block_process(int bi, int bj) {
        int n = bs * nb;
        double dm = 0;
        for (int i = 1 + bi * bs; i <= min((bi + 1) * bs, n); ++i) {
            for (int j = 1 + bj * bs; j <= min((bj + 1) * bs, n); ++j) {
                double tmp = u[i][j];
                u[i][j] = 0.25 * abs(u[i - 1][j] + u[i + 1][j] +
                                    u[i][j - 1] + u[i][j + 1] -
                                    h * h * f[i][j]);
                dm = max(dm, abs(tmp - u[i][j]));
            }
        }
        return dm;
    }

public:
    enum start_value { ZERO = 0, RAND = 1, AVERAGE = 2 };

    approximator(function<double(double, double)> &fun_g,
                function<double(double, double)> &fun_f,
                int block_size, int num_block, double eps, start_value sv) :
    fun(fun_g), bs(block_size), nb(num_block), h(1.0 / (bs * nb + 1)), eps(eps) {
        int n = bs * nb;
        u = vector<vector<double>>(n + 2, vector<double>(n + 2, 0.0));
        f = vector<vector<double>>(n + 2, vector<double>(n + 2, 0.0));
        
        double lower = 1e9;
        double upper = -1e9;
        double avg = 0.0;
        for (int i = 0; i < n + 2; ++i) {
            u[i][0] = fun_g(i*h, 0);
            u[i][n + 1] = fun_g(i*h, (n + 1) * h);
            u[0][i] = fun_g(0, i*h);
            u[n + 1][i] = fun_g((n + 1) * h, i*h);
            
            avg += u[i][0] + u[0][i] + u[i][n + 1] + u[n + 1][i];
            lower = min(lower, u[i][0]);
            lower = min(lower, u[0][i]);
            lower = min(lower, u[i][n + 1]);
            lower = min(lower, u[n + 1][i]);

            upper = max(upper, u[i][0]);
            upper = max(upper, u[0][i]);
            upper = max(upper, u[i][n + 1]);
            upper = max(upper, u[n + 1][i]);
        }

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                f[i][j] = fun_f(i * h, j * h);
            }
        }

        switch (sv)
        {
        case AVERAGE:
            avg = (avg - u[0][0] - u[0][n + 1] - u[n + 1][0] - u[n + 1][n + 1]) / (4 * (n + 1));

            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    u[i][j] = avg;
                }
            }
            break;

        case RAND:    
            uniform_real_distribution<double> unif(lower, upper);
            default_random_engine re;

            for (int i = 1; i < n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    u[i][j] = unif(re);
                }
            }
            break;
        }
    }

    void calc_approximation() {
        vector<double> dm(nb, 0);
        double dmax;

        do {
            dmax = 0.0;
            dm.assign(nb, 0.0);
            for (int nx = 0; nx < nb; ++nx) {
                int i, j;
#pragma omp parallel for shared(nx, dm, u) private(i, j)
                for (i = 0; i <= nx; ++i) {
                    j = nx - i;
                    double d = block_process(i, j);
                    dm[i] = max(dm[i], d);
                }
            }
            for (int nx = nb - 2; nx >= 0; --nx) {
                int i, j;
#pragma omp parallel for shared(nx, dm, u) private(i, j)
                for (i = nb - nx - 1; i < nb; ++i) {
                    j = 2 * (nb - 1) - nx - i;
                    double d = block_process(i, j);
                    dm[i] = max(dm[i], d);
                }
            }
            for (int i = 0; i < nb; ++i) {
                dmax = max(dmax, dm[i]);
            }
        } while (dmax > eps);
    }

    double calc_err() {
        int n = bs * nb;
        double sum_err = 0.0;
        int num_zero = 0;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                double expected = fun(i * h, j * h);
                if (abs(expected) > zero) {
                    sum_err += abs(u[i][j] - expected) / abs(expected);
                } else {
                    ++num_zero;
                }
            }
        }
        return sum_err / (n * n - num_zero);
    }
};

struct result {
    double time;
    double error;
};

result experiment(int num_thread,
                            double eps,
                            function<double(double, double)> &fun_g,
                            function<double(double, double)> &fun_f,
                            int block_size,
                            int num_block,
                            approximator::start_value sv) {
    omp_set_num_threads(num_thread);
    approximator appr(fun_g, fun_f, block_size, num_block, eps, sv);
    auto start_time = omp_get_wtime();
    appr.calc_approximation();
    auto end_time = omp_get_wtime();
    return {end_time - start_time, appr.calc_err()};
}

// functions

// function<double(double, double)> fun_g = [](double x, double y) { return 52.0; };
// function<double(double, double)> fun_f = [](double x, double y) { return 0.0; };

// function<double(double, double)> fun_g = [](double x, double y) { return 2 * x + 6 * y; };
// function<double(double, double)> fun_f = [](double x, double y) { return 0.0; };

// function<double(double, double)> fun_g = [](double x, double y) { return 2 * pow(x, 3) + 6 * pow(y, 3); };
// function<double(double, double)> fun_f = [](double x, double y) { return 12 * x + 36 * y; };

// function<double(double, double)> fun_g = [](double x, double y) { return sin(x) - cos(y); };
// function<double(double, double)> fun_f = [](double x, double y) { return -sin(x) + cos(y); };

int main() {
    cout << "sorry I'm a cat\n";
    return 0;
}