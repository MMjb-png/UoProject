#ifndef BATCH_QUEUE_HPP
#define BATCH_QUEUE_HPP

#include <vector>
#include <mutex>
#include <condition_variable>
#include <torch/torch.h>
// GoBoard.hpp をインクルード
#include "board/GoBoard.hpp"
#include "board/BoardData.hpp" // ← これを追加して game_info_t を認識させる

struct BatchItem {
    at::Tensor input;     // NN入力特徴量
    game_info_t *game;    // 局面情報（ProcessMiniBatchで展開に使用）
    int node_index;       // 親ノードのインデックス
    int child_index;      // 選択した子のインデックス
    int color;            // 手番
};

class BatchQueue {
private:
    std::vector<BatchItem> queue;
    std::mutex mtx;
    std::condition_variable cv;
    const size_t max_batch_size;

public:
    BatchQueue(size_t batch_size = 1) : max_batch_size(batch_size) {}

    // ワーカースレッドが使用：推論リクエストを積む
    void push(at::Tensor input, game_info_t *game, int node, int child, int color) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push_back({input, game, node, child, color});
        }
        // メインスレッドに「データが入った」ことを通知
        cv.notify_all();
    }

    // メインスレッドが使用：溜まっているデータを一括取得
    std::vector<BatchItem> pop_all() {
        std::unique_lock<std::mutex> lock(mtx);
        // データが来るまで待機（タイムアウト付きでデッドロック防止）
        cv.wait_for(lock, std::chrono::milliseconds(10), [this] { return !queue.empty(); });
        
        std::vector<BatchItem> items = std::move(queue);
        queue.clear();
        return items;
    }

    // メインスレッドが使用：推論完了を全ワーカースレッドに通知
    void notify_completion() {
        cv.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};

#endif