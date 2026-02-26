#ifndef BATCH_QUEUE_HPP
#define BATCH_QUEUE_HPP

#include <vector>
#include <mutex>
#include <condition_variable>
#include <utility> // std::pair用
#include <torch/torch.h>
#include "board/GoBoard.hpp"
#include "board/BoardData.hpp"

/**
 * @brief 推論リクエスト1回分のデータを保持する構造体
 */
struct BatchItem {
    at::Tensor input;
    game_info_t *game;
    // これが定義されている必要があります！
    std::vector<std::pair<int, int>> path; 
    int color;
};

/**
 * @brief ワーカースレッドとメインスレッド（BatchHandler）を繋ぐキュー
 */
class BatchQueue {
private:
    std::vector<BatchItem> queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool is_interrupted;

public:
    BatchQueue() : is_interrupted(false) {}

    /**
     * @brief ワーカースレッドが使用：推論待ちの経路と局面を積む
     */
    void push(at::Tensor input, game_info_t *game, const std::vector<std::pair<int, int>>& path, int color) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push_back({input, game, path, color});
        }
        // メインスレッド（BatchHandler）にデータが入ったことを通知
        cv.notify_all();
    }

    /**
     * @brief BatchHandlerが使用：溜まっているリクエストを一括で取り出す
     */
    std::vector<BatchItem> pop_all() {
        std::unique_lock<std::mutex> lock(mtx);
        
        // データが来るまで最大10ms待機（デッドロック防止）
        cv.wait_for(lock, std::chrono::milliseconds(10), [this] { 
            return !queue.empty() || is_interrupted; 
        });
        
        if (queue.empty()) return {};

        std::vector<BatchItem> items = std::move(queue);
        queue.clear();
        return items;
    }

    /**
     * @brief 推論完了を全スレッドに通知する
     */
    void notify_completion() {
        cv.notify_all();
    }

    /**
     * @brief キューのサイズ確認
     */
    size_t size() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    /**
     * @brief 強制停止（探索終了時にスレッドを解放する）
     */
    void interrupt() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            is_interrupted = true;
        }
        cv.notify_all();
    }
};

#endif