#include "mcts/BatchQueue.hpp"
#include "mcts/UctSearch.hpp"
#include <torch/torch.h>
#include <vector>

// --- 追加: ExpandNodeWithNN の宣言 ---
// UctSearch.cpp で定義されているこの関数を BatchHandler から使えるようにします
extern int ExpandNodeWithNN(game_info_t *game, int color, int current, int child_index, at::Tensor policy);
// ------------------------------------

extern uct_node_t *uct_node;
extern BatchQueue g_batch_queue;

// 外部で定義されているグローバル変数を参照
extern uct_node_t *uct_node;
extern BatchQueue g_batch_queue;

/**
 * @brief 勝利数(win)をスレッドセーフに加算する内部関数
 * doubleのAtomic操作をCASループで実現
 */
static void AddWinCount(child_node_t &child, double v) {
    auto& atomic_win = reinterpret_cast<std::atomic<double>&>(child.win);
    double old_win = atomic_win.load();
    while (!atomic_win.compare_exchange_weak(old_win, old_win + v));
}

/**
 * @brief メインスレッドから呼び出され、溜まったキューを一括で推論・反映する
 */
void ProcessMiniBatch(torch::jit::script::Module &model, torch::Device &device) {
    // 1. キューから全てのアイテムを取り出す
    auto items = g_batch_queue.pop_all();
    if (items.empty()) return;

    int batch_size = items.size();
    std::vector<torch::Tensor> tensor_list;
    for (const auto& item : items) {
        tensor_list.push_back(item.input);
    }

    try {
        // 2. テンソルを結合してGPU/CPUへ転送
        auto batched_tensor = torch::stack(tensor_list).to(device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batched_tensor);

        // 3. ネットワーク実行 (Policy と Value を取得)
        auto outputs = model.forward(inputs).toTuple();
        auto policy_out = outputs->elements()[0].toTensor().softmax(1).to(torch::kCPU);
        auto value_out = outputs->elements()[1].toTensor().to(torch::kCPU);

        // 4. 結果を各探索経路の全ノードに反映
        for (int i = 0; i < batch_size; ++i) {
            auto& item = items[i];
            float win_rate = value_out[i].item<float>(); // NNの評価値 (Value)
            auto& path = item.path;

            if (path.empty()) continue;

            // --- TamaGo完全再現：Pathを逆順に遡って更新 ---
            float v = win_rate;
            for (int j = path.size() - 1; j >= 0; --j) {
                int node_idx = path[j].first;
                int child_idx = path[j].second;
                child_node_t &child = uct_node[node_idx].child[child_idx];

                // 訪問回数の加算 (Atomic)
                child.move_count.fetch_add(1);
                // 勝利数の加算 (Atomic)
                AddWinCount(child, static_cast<double>(v));
                
                // 親ノード全体の訪問数も更新
                uct_node[node_idx].move_count.fetch_add(1);

                // 手番が入れ替わるため、勝率を反転 (1.0 - v)
                v = 1.0f - v;
            }

            // 5. 展開処理（葉ノードにPolicyをセットする）
            // 探索で最後に到達したノードに対して展開を行う
            auto last_step = path.back();
            int leaf_node_idx = uct_node[last_step.first].child[last_step.second].index;
            
            // ノードが未展開(NOT_EXPANDED)の場合に展開を実行
            if (leaf_node_idx == NOT_EXPANDED) {
                // ExpandNode は内部で uct_node[last_step.first].child[last_step.second].index を更新する想定
                ExpandNodeWithNN(item.game, item.color, last_step.first, last_step.second, policy_out[i]);
            }

            // 6. Virtual Loss の解除
            // 最初に加算した「探索中ペナルティ」をここで解除する
            uct_node[last_step.first].child[last_step.second].virtual_loss.fetch_sub(1);

            // 7. メモリ解放
            if (item.game) {
                FreeGame(item.game);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in ProcessMiniBatch: " << e.what() << std::endl;
    }

    // 8. 完了を全スレッドに通知
    g_batch_queue.notify_completion();
}