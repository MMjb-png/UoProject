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

void ProcessMiniBatch(torch::jit::script::Module &model, torch::Device &device) {
    auto items = g_batch_queue.pop_all();
    if (items.empty()) return;

    torch::DeviceGuard device_guard(device);

    // 安定するまでバッチサイズを絞る（例: 最大 16）
    if (items.size() > 16) {
        for (size_t i = 16; i < items.size(); ++i) {
            if (items[i].game) FreeGame(items[i].game);
        }
        items.erase(items.begin() + 16, items.end());
    }

    int batch_size = items.size();
    
    // ★ 勾配計算を無効化してメモリ節約
    torch::NoGradGuard no_grad;

    try {
        std::vector<torch::Tensor> tensor_list;
        for (auto& item : items) {
            tensor_list.push_back(item.input);
        }

        auto batched_tensor = torch::stack(tensor_list).to(device);
        auto input_reshaped = batched_tensor.view({-1, 6, 9, 9});

        // ★ 推論実行
        auto outputs = model.forward({input_reshaped}).toTuple();
        
        auto policy_out = outputs->elements()[0].toTensor().softmax(1).to(torch::kCPU);
        auto value_out = outputs->elements()[1].toTensor().to(torch::kCPU);

        for (int i = 0; i < batch_size; ++i) {
            auto& item = items[i];
            
            // Value計算
            auto v_probs = torch::softmax(value_out[i], 0); 
            float v = (item.color == S_BLACK) ? 
                      (v_probs[0].item<float>() + v_probs[1].item<float>()*0.5f) : 
                      (v_probs[2].item<float>() + v_probs[1].item<float>()*0.5f);

            // --- 1. バックプロパゲーション ---
            float current_v = v; 
            for (int j = (int)item.path.size() - 1; j >= 0; --j) {
                int node_idx = item.path[j].first;
                int child_idx = item.path[j].second;
                child_node_t &child = uct_node[node_idx].child[child_idx];

                child.move_count.fetch_add(1);
                AddWinCount(child, static_cast<double>(current_v));
                uct_node[node_idx].move_count.fetch_add(1);

                current_v = 1.0f - current_v;
            }

            // 展開処理
            auto last_step = item.path.back();
            ExpandNodeWithNN(item.game, item.color, last_step.first, last_step.second, policy_out[i]);

            if (item.game) { FreeGame(item.game); item.game = nullptr; }
        }
    } catch (const std::exception& e) {
        std::cerr << "LibTorch Error: " << e.what() << std::endl;
    }

    g_batch_queue.notify_completion();
}