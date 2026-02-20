#ifndef TAMAGO_FEATURE_HPP
#define TAMAGO_FEATURE_HPP

#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <string>
#include "board/GoBoard.hpp"

class TamaGoFeature {
public:
    static std::unordered_map<int, int> CreateSymmetryMap(int board_size, int sym);
    static int GetSymmetricalCoordinate(int pos, int board_size, int sym);
    static std::vector<int> GetBoardData(const game_info_t *game, int sym, int color);

    static int ConvertFromGtpFormat(const std::string& gtp, int board_size);
    static std::string ConvertToGtpFormat(int pos, int board_size);

    static torch::Tensor GenerateInputPlanes(const game_info_t *game, int color);

    // 引数2つの Extract (cpp 82行目: at::Tensor を返す)
    static torch::Tensor Extract(const game_info_t *game, int color);

    // 引数3つの Extract (もし cpp にあるなら)
    static torch::Tensor Extract(const game_info_t *game, int color, int sym);

    // 【重要】戻り値を std::vector<float> に修正 (cpp 88行目と一致させる)
    static std::vector<float> GenerateRLTargetData(const std::string &improved_policy_data, int sym);

    // 他のターゲット生成関数 (戻り値が torch::Tensor のもの)
    static torch::Tensor GenerateTargetData(const game_info_t *game, int target_pos, int sym);
    static torch::Tensor GenerateRLTargetData(const game_info_t *game, const std::string& improved_policy, int sym);
};

#endif