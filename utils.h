#pragma once

#include <vector>
#include <random>

template<typename T>
std::vector<T> random_vector(size_t N) {
    std::vector<T> v(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);

    for (size_t i = 0; i < N; i++) {
        v[i] = dis(gen);
    }

    return v;
}