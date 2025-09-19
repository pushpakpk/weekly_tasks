# Task3/benchmark.py
import numpy as np
import time
import csv
import psutil
from Task3.timeseries_utils import (rolling_mean_numpy_1d,
                                    rolling_mean_pandas_1d,
                                    ewma_numpy, fft_bandpass_1d)

def mem_rss_mb():
    p = psutil.Process()
    return p.memory_info().rss / (1024*1024)

def benchmark_one(n_rows=1_200_000, window=100, runs=3):
    print("Generating data...")
    x = np.random.normal(size=n_rows).astype(np.float64)

    results = []
    # rolling numpy
    for _ in range(runs):
        t0 = time.perf_counter(); m0 = mem_rss_mb()
        y = rolling_mean_numpy_1d(x, window)
        t1 = time.perf_counter(); m1 = mem_rss_mb()
        results.append(("numpy_rolling", n_rows, t1-t0, m1-m0))
    # rolling pandas
    for _ in range(runs):
        t0 = time.perf_counter(); m0 = mem_rss_mb()
        y = rolling_mean_pandas_1d(x, window)
        t1 = time.perf_counter(); m1 = mem_rss_mb()
        results.append(("pandas_rolling", n_rows, t1-t0, m1-m0))

    # ewma
    for _ in range(runs):
        t0 = time.perf_counter(); m0 = mem_rss_mb()
        y = ewma_numpy(x, alpha=0.01)
        t1 = time.perf_counter(); m1 = mem_rss_mb()
        results.append(("ewma_numpy", n_rows, t1-t0, m1-m0))

    # fft bandpass small test
    for _ in range(runs):
        t0 = time.perf_counter(); m0 = mem_rss_mb()
        y = fft_bandpass_1d(x[:100000], low_freq=0.01, high_freq=0.1, fs=1.0)
        t1 = time.perf_counter(); m1 = mem_rss_mb()
        results.append(("fft_bandpass_100k", 100000, t1-t0, m1-m0))

    # write results
    with open("Task3/benchmark_results.csv", "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(['method','n_rows','time_sec','mem_mb'])
        for row in results:
            w.writerow(row)

    print("Done. Results saved to Task3/benchmark_results.csv")

if __name__ == "__main__":
    benchmark_one()
