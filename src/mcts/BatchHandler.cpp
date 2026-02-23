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
        // 2. テンソルを結合
        auto batched_tensor = torch::stack(tensor_list).to(device);

        // --- 重要：次元の修正 ---
        // stack後の [batch, 1, 6, 9, 9] を [batch, 6, 9, 9] に落とす
        // もしくは view({-1, 6, 9, 9}) で強制的に形を固定します
        auto input_reshaped = batched_tensor.view({-1, 6, 9, 9});

        // 3. ネットワーク実行
        // 変数名を 'model' に合わせ、入力に input_reshaped を使用
        auto outputs = model.forward({input_reshaped}).toTuple();
        
        auto policy_out = outputs->elements()[0].toTensor().softmax(1).to(torch::kCPU);
        auto value_out = outputs->elements()[1].toTensor().to(torch::kCPU);

        // 4. 結果を各探索経路の全ノードに反映
        // 4. 結果を各探索経路の全ノードに反映
        for (int i = 0; i < batch_size; ++i) {
            auto& item = items[i];
            
            // --- TamaGo Value Head の解析に基づく勝率計算 ---
            // value_out[i] は [logit_win, logit_draw, logit_loss] の 3要素
            auto v_probs = torch::softmax(value_out[i], 0); 

            float p_black_win = v_probs[0].item<float>();
            float p_draw      = v_probs[1].item<float>();
            float p_white_win = v_probs[2].item<float>();

            // 黒から見た期待値 (Win=1.0, Draw=0.5, Loss=0.0)
            float win_rate_as_black = p_black_win + (p_draw * 0.5f);

            // 現在の手番 (item.color) から見た勝率 v を計算
            float v = (item.color == S_BLACK) ? win_rate_as_black : (1.0f - win_rate_as_black);

            // 勝率計算の直後に挿入
            fprintf(stderr, "DEBUG_V: win=%.3f draw=%.3f loss=%.3f -> v=%.3f\n", p_black_win, p_draw, p_white_win, v);

            auto& path = item.path;
            if (path.empty()) continue;

            // --- バックプロパゲーション (Pathを逆順に遡って更新) ---
            float current_v = v; 
            for (int j = (int)path.size() - 1; j >= 0; --j) {
                int node_idx = path[j].first;
                int child_idx = path[j].second;
                child_node_t &child = uct_node[node_idx].child[child_idx];

                // 訪問回数と勝利数の加算
                child.move_count.fetch_add(1);
                AddWinCount(child, static_cast<double>(current_v));
                
                // 親ノード全体の訪問数も更新
                uct_node[node_idx].move_count.fetch_add(1);

                // 手番が入れ替わるため、次（親方向）の評価値は反転
                current_v = 1.0f - current_v;
            }

            // 5. 展開処理（葉ノードにPolicyをセットする）
            auto last_step = path.back();
            // ProcessMiniBatch のループ内（ステップ 5, 6 のあたり）
            ExpandNodeWithNN(item.game, item.color, last_step.first, last_step.second, policy_out[i]);

            // 6. Virtual Loss の解除
            uct_node[last_step.first].child[last_step.second].virtual_loss.fetch_sub(1);

            // 【重要】ここで安全に解放
            if (item.game != nullptr) {
                FreeGame(item.game);
                // 二重解放防止のため、念のため nullptr を入れる（itemがコピーなら意味はないが安全策）
            }
            // item.game は ExpandNodeWithNN 内部で FreeGame されるため、ここでは不要
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in ProcessMiniBatch: " << e.what() << std::endl;
    }

    // 8. 完了を全スレッドに通知
    g_batch_queue.notify_completion();
}