#include "mcts/BatchQueue.hpp"
#include "mcts/UctSearch.hpp"
#include <torch/torch.h>
#include <torch/script.h> 
#include <vector>

// 外部で定義されているグローバル変数を参照
extern uct_node_t *uct_node;
extern BatchQueue g_batch_queue;

extern torch::Device device;             // 外部参照宣言

extern torch::jit::script::Module tamago_model;

// --- 追加: ExpandNodeWithNN の宣言 ---
// UctSearch.cpp で定義されているこの関数を BatchHandler から使えるようにします
extern int ExpandNodeWithNN(game_info_t *game, int color, int current, int child_index, torch::Tensor policy_tensor);

extern void SetNodeStatistics(int node_index, const torch::Tensor& policy_tensor, double value);

extern void UpdateNodeStats(int node_idx, int child_idx, double v);

static void AddWinCount(child_node_t &child, double v) {
    // child.win が std::atomic<double> の場合
    double old_win = child.win.load();
    // compare_exchange_weak を用いて安全に加算
    while (!child.win.compare_exchange_weak(old_win, old_win + v));
}

void ProcessMiniBatch(torch::jit::script::Module& model, torch::Device& device) {
    auto items = g_batch_queue.pop_all();
    if (items.empty()) return;

    try {
        std::vector<torch::Tensor> tensor_list;
        for (auto& item : items) {
            tensor_list.push_back(item.input);
        }
        
        torch::Tensor inputs = torch::cat(tensor_list, 0).to(device);
        
        torch::NoGradGuard no_grad; 
        model.eval(); 

        auto outputs_ivalue = model.forward({inputs});
        auto output_tuple = outputs_ivalue.toTuple();
        
        torch::Tensor policy_out = output_tuple->elements()[0].toTensor().to(torch::kCPU);
        torch::Tensor value_out = output_tuple->elements()[1].toTensor().to(torch::kCPU);

        for (size_t i = 0; i < items.size(); ++i) {
            auto& item = items[i];
            
            // NNの出力を「黒番の勝率」として取得
            auto v_probs = torch::softmax(value_out[i], 0); 
            float* p_v = v_probs.data_ptr<float>();
            double v_black = static_cast<double>(p_v[0] * 1.0f + p_v[1] * 0.5f);

            // ★修正：そのノードの手番に合わせて「自分視点」の勝率に変換
            double v_node_perspective = (item.color == S_BLACK) ? v_black : (1.0 - v_black);

            // ノード展開
            int idx = ExpandNodeWithNN(&item.game_copy, item.color, -1, -1, policy_out[i]);

            if (idx != -1) {
                // そのノードの手番プレイヤーにとっての評価値をセット
                uct_node[idx].win = v_node_perspective; 
                uct_node[idx].width = 1; 

                // デバッグ用
                if (idx < 50) { // ルート付近のみ
                    std::cerr << "NodeIdx: " << idx << " Color: " << item.color 
                            << " Val(Black): " << v_black 
                            << " Val(Node): " << v_node_perspective << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        // リカバリが必要な場合は、ここでも何らかの値をセットしてスレッドを戻す
    }
    
    g_batch_queue.notify_completion();
}