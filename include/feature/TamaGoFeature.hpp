#include <torch/torch.h>
#include "board/GoBoard.hpp"

class TamaGoFeature {
public:
    static torch::Tensor Extract(const game_info_t *game, int color) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        // [1, 6, 9, 9] のテンソルを作成 (Python版の in_channels=6 に合わせる)
        auto input = torch::zeros({1, 6, pure_board_size, pure_board_size}, options);
        
        for (int i = 0; i < pure_board_max; i++) {
            int pos = onboard_pos[i];
            int x = i % pure_board_size;
            int y = i / pure_board_size;
            
            // 1: 自分の石, 2: 相手の石
            if (game->board[pos] == color) input[0][0][y][x] = 1.0f;
            else if (game->board[pos] == GetOppositeColor(color)) input[0][1][y][x] = 1.0f;
            
            // 3: 1手前
            if (pos == game->record[game->moves - 1].pos) input[0][2][y][x] = 1.0f;
            // 4: 2手前
            if (game->moves > 1 && pos == game->record[game->moves - 2].pos) input[0][3][y][x] = 1.0f;
            // 5: 空き地
            if (game->board[pos] == S_EMPTY) input[0][4][y][x] = 1.0f;
            // 6: 手番色 (黒なら1.0)
            input[0][5][y][x] = (color == S_BLACK ? 1.0f : 0.0f);
        }
        return input;
    }
};