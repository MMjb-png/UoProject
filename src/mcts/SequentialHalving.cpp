#include "mcts/SequentialHalving.hpp"
#include <cmath>
#include <algorithm>

std::vector<int> GetSequenceOfConsideredVisits(int max_considered_actions, int num_simulations) {
    if (max_considered_actions <= 1) {
        std::vector<int> seq(num_simulations, 0);
        return seq;
    }

    int log2max = static_cast<int>(std::ceil(std::log2(max_considered_actions)));
    std::vector<int> sequence;
    std::vector<int> visits(max_considered_actions, 0);
    int num_considered = max_considered_actions;

    while (sequence.size() < (size_t)num_simulations) {
        int num_extra_visits = std::max(1, num_simulations / (log2max * num_considered));
        for (int i = 0; i < num_extra_visits; ++i) {
            for (int j = 0; j < num_considered; ++j) {
                sequence.push_back(num_considered); // 現在の探索幅を記録
                visits[j]++;
            }
        }
        num_considered = std::max(2, num_considered / 2);
    }
    
    if (sequence.size() > (size_t)num_simulations) {
        sequence.resize(num_simulations);
    }
    return sequence;
}