#include "mcts/BatchQueue.hpp"
#include "mcts/UctSearch.hpp"
#include <torch/torch.h>
#include <vector>

// --- 追加: ExpandNodeWithNN の宣言 ---
// UctSearch.cpp で定義されているこの関数を BatchHandler から使えるようにします
extern int ExpandNodeWithNN(game_info_t *game, int color, int current, int child_index, at::Tensor policy);
// ------------------------------------

// 外部で定義されているグローバル変数を参照
extern uct_node_t *uct_node;
extern BatchQueue g_batch_queue;

/**
 * @brief 勝利数(win)をスレッドセーフに加算する内部関数
 * doubleのAtomic操作をCASループで実現
 */
static void AddWinCount(child_node_t &child, double v) {
    // child.win が double の場合
    auto* atomic_ptr = reinterpret_cast<std::atomic<double>*>(&child.win);
    double old_win = atomic_ptr->load();
    while (!atomic_ptr->compare_exchange_weak(old_win, old_win + v));
}

/**
 * @brief メインスレッドから呼び出され、溜まったキューを一括で推論・反映する
 */
/**
 * @brief メインスレッドから呼び出され、溜まったキューを一括で推論・反映する
 */
void ProcessMiniBatch(torch::jit::script::Module &model, torch::Device &device) {
    auto items = g_batch_queue.pop_all();
    if (items.empty()) return;

    int batch_size = items.size();
    std::vector<torch::Tensor> tensor_list;
    for (const auto& item : items) {
        tensor_list.push_back(item.input);
    }

    try {
        auto batched_tensor = torch::stack(tensor_list).to(device);
        auto input_reshaped = batched_tensor.view({-1, 6, 9, 9});

        auto outputs = model.forward({input_reshaped}).toTuple();
        
        auto policy_out = outputs->elements()[0].toTensor().softmax(1).to(torch::kCPU);
        auto value_out = outputs->elements()[1].toTensor().to(torch::kCPU);

        for (int i = 0; i < batch_size; ++i) {
            auto& item = items[i];
            
            // TamaGo Value Head: [Win, Draw, Loss]
            auto v_probs = torch::softmax(value_out[i], 0); 
            float p_black_win = v_probs[0].item<float>();
            float p_draw      = v_probs[1].item<float>();
            float p_white_win = v_probs[2].item<float>();

            // 黒から見た期待値計算
            float win_rate_as_black = p_black_win + (p_draw * 0.5f);
            // 現在の手番(item.color)から見た勝率を求める
            float v = (item.color == S_BLACK) ? win_rate_as_black : (1.0f - win_rate_as_black);

            fprintf(stderr, "DEBUG_V: B_win=%.3f draw=%.3f W_win=%.3f -> v=%.3f (color=%d)\n", 
                    p_black_win, p_draw, p_white_win, v, item.color);

            auto& path = item.path;
            if (path.empty()) {
                if (item.game) FreeGame(item.game);
                continue;
            }

            // --- 1. バックプロパゲーション (Pathを逆順に遡って更新) ---
            float current_v = v; 
            for (int j = (int)path.size() - 1; j >= 0; --j) {
                int node_idx = path[j].first;
                int child_idx = path[j].second;
                child_node_t &child = uct_node[node_idx].child[child_idx];

                // 子ノードの統計更新
                child.move_count.fetch_add(1);
                AddWinCount(child, static_cast<double>(current_v)); // 前回のUpdateNodeStatsと同様のAtomic加算
                
                // 親ノード全体の訪問数更新
                uct_node[node_idx].move_count.fetch_add(1);

                // 親から見た価値に反転
                current_v = 1.0f - current_v;
            }

            // --- 2. 展開処理（葉ノードにPolicyをセット） ---
            auto last_step = path.back();
            // ここで child[last_step.second].index が NOT_EXPANDED から新しいノード番号へ更新される
            ExpandNodeWithNN(item.game, item.color, last_step.first, last_step.second, policy_out[i]);

            // --- 3. Virtual Loss の解除 ---
            // UctSearch で加算された仮想損失をここで引く
            uct_node[last_step.first].child[last_step.second].virtual_loss.fetch_sub(1);

            // --- 4. メモリ解放 ---
            if (item.game != nullptr) {
                FreeGame(item.game);
                item.game = nullptr;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in ProcessMiniBatch: " << e.what() << std::endl;
    }

    g_batch_queue.notify_completion();
}