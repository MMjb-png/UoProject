/**
 * @file mcts/SequentialHalving.hpp
 * @brief Sequential Halving for Gumbel AlphaZero
 */
#pragma once

#include <vector>
#include <unordered_map>

/**
 * @brief 探索回数に対応する閾値列を取得
 * @param max_considered_actions 探索幅の最大値
 * @param num_simulations 1回の思考で実行する探索回数
 * @return 探索回数閾値の列
 */
std::vector<int> GetSequenceOfConsideredVisits(int max_considered_actions, int num_simulations);

/**
 * @brief 探索幅と探索回数のペアを取得
 * @param max_considered_actions 探索幅の最大値
 * @param num_simulations 1回の思考で実行する探索回数
 * @return 探索幅をキー、探索回数をバリューに持つ辞書
 */
std::unordered_map<int, int> GetCandidatesAndVisitPairs(int max_considered_actions, int num_simulations);