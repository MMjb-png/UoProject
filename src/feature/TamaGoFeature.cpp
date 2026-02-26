#pragma push_macro("X")
#pragma push_macro("Y")
#undef X
#undef Y

#include "feature/TamaGoFeature.hpp"
#include "board/Point.hpp"
#include "board/Constant.hpp"
#include "util/Utility.hpp"
#include <cmath>
#include <sstream>

#pragma pop_macro("X")
#pragma pop_macro("Y")

extern int pure_board_size;

// 対称形変換テーブルの生成
std::unordered_map<int, int>
TamaGoFeature::CreateSymmetryMap(int board_size, int sym)
{
    std::unordered_map<int, int> sym_map;
    for (int y = 1; y <= board_size; y++) {
        for (int x = 1; x <= board_size; x++) {
            int original_pos = POS(x, y);
            int nx = x - 1, ny = y - 1;
            int tx, ty;

            // sym (0-7) に応じた座標変換
            if (sym < 4) { // 回転
                for (int i = 0; i < sym; i++) {
                    int tmp = nx; nx = ny; ny = board_size - 1 - tmp;
                }
            } else { // 反転 + 回転
                nx = board_size - 1 - nx;
                for (int i = 0; i < (sym - 4); i++) {
                    int tmp = nx; nx = ny; ny = board_size - 1 - tmp;
                }
            }
            sym_map[original_pos] = POS(nx + 1, ny + 1);
        }
    }
    return sym_map;
}

torch::Tensor
TamaGoFeature::GenerateInputPlanes(const game_info_t *game, int color)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int size = pure_board_size; // 9
    auto input = torch::zeros({1, 6, size, size}, options);
    auto in_acc = input.accessor<float, 4>();
    int opp_color = (color == S_BLACK) ? S_WHITE : S_BLACK;

    // --- 盤面情報の抽出 (onboard_pos基準) ---
    for (int i = 0; i < size * size; i++) {
        int pos = onboard_pos[i]; // Rayの1次元座標
        int stone = game->board[pos];

        // テンソル上の座標 (0-indexed)
        int tx = i % size;
        int ty = i / size;

        // 1-3枚目: 盤面状態
        if (stone == S_EMPTY) {
            in_acc[0][0][ty][tx] = 1.0f;
        } else if (stone == color) {
            in_acc[0][1][ty][tx] = 1.0f;
        } else if (stone == opp_color) {
            in_acc[0][2][ty][tx] = 1.0f;
        }

        // 4枚目: 直前の着手履歴 (Movesが2手目以降の場合のみ)
        if (game->moves > 1) {
            int prev_pos = game->record[game->moves - 1].pos;
            if (prev_pos != PASS && pos == prev_pos) {
                in_acc[0][3][ty][tx] = 1.0f;
            }
        }
    }

    // --- 5枚目: パスフラグ (P[4]) ---
    // 初手(Moves=1以下)での誤検知を完全にブロック
    if (game->moves > 1) {
        int last_move_pos = game->record[game->moves - 1].pos;
        if (last_move_pos == PASS) {
            for (int r = 0; r < size; r++)
                for (int c = 0; c < size; c++) in_acc[0][4][r][c] = 1.0f;
        }
    }

    // --- 6枚目: 手番の色 ---
    float turn_val = (color == S_BLACK) ? 1.0f : -1.0f;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            in_acc[0][5][r][c] = turn_val;
        }
    }

    return input;
}

torch::Tensor
TamaGoFeature::Extract(const game_info_t *game, int color)
{
    return GenerateInputPlanes(game, color);
}

std::vector<float>
TamaGoFeature::GenerateRLTargetData(const std::string &improved_policy_data, int sym)
{
    // 強化学習の教師ラベル生成ロジック (略)
    std::vector<float> target(pure_board_size * pure_board_size + 1, 0.0f);
    return target;
}