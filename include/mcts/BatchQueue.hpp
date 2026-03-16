#ifndef TAMA_BATCH_QUEUE_HPP
#define TAMA_BATCH_QUEUE_HPP

#include <torch/torch.h>
#include <vector>
#include <mutex>
#include <utility>
#include "board/GoBoard.hpp"

/**
 * @brief 推論待ちの1局面分のデータを保持する構造体
 */
struct BatchItem {
    torch::Tensor input;
    game_info_t game_copy; // ★ポインタではなく実体
    std::vector<std::pair<int, int>> path;
    int color;

    // コンストラクタ（引数は4つ）
    BatchItem(torch::Tensor i, const game_info_t *g, const std::vector<std::pair<int, int>>& p, int c)
        : input(i), path(p), color(c) {
        if (g) {
            game_copy = *g; // ここで中身をコピー（寿命問題を解決）
        }
    }
};
/**
 * @brief 推論待ちキューを管理するクラス
 * 複数スレッドからの push と、推論スレッドによる一括取り出し(pop_all)を安全に行います。
 */
class BatchQueue {
private:
    std::vector<BatchItem> queue_items; // キューの実体
    std::mutex mtx;                     // スレッドセーフのためのミューテックス

public:
    BatchQueue() = default;
    /**
     * @brief キューにデータを追加する
     * @return 追加後のキューのサイズ
     */
    size_t push(torch::Tensor input, game_info_t *game, const std::vector<std::pair<int, int>>& path, int color) {
        if (game == nullptr) {
            std::cerr << "ERROR: game pointer is NULL in push!" << std::endl;
            return queue_items.size();
        }

        // BatchItem のコンストラクタに合わせて引数を 4 つで呼び出す
        queue_items.emplace_back(input, game, path, color);

        std::cerr << "Debug: Inside push - added to queue. size: " << queue_items.size() << std::endl;
        return queue_items.size();
    }

    /**
     * @brief 現在キューに入っているすべてのアイテムを取り出し、キューを空にする
     * @return 取り出したアイテムのリスト
     */
    std::vector<BatchItem> pop_all() {
        std::vector<BatchItem> extracted = std::move(queue_items);
        queue_items.clear();
        return extracted;
    }

    /**
     * @brief キューが空かどうかを確認する
     */
    bool empty() {
        return queue_items.empty();
    }

    /**
     * @brief 現在のキューのサイズを取得する
     */
    size_t size() {
        return queue_items.size();
    }

    void notify_completion() {
        // ここでは特に何もしませんが、必要に応じて条件変数などで待機スレッドを通知することもできます。
    }
};

// グローバルなバッチキューの宣言（実体は cpp ファイル等で定義）
extern BatchQueue g_batch_queue;

#endif // TAMA_BATCH_QUEUE_HPP