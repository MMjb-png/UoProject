/**
 * @file src/mcts/UctSearch.cpp
 * @author Yuki Kobayashi
 * @~english
 * @brief Monte-Carlo tree search with upper confidence bound.
 * @~japanese
 * @brief UCBを利用したモンテカルロ木探索
 */

#include "feature/TamaGoFeature.hpp"  // <--- これを追加
#include <torch/script.h>             // <--- LibTorchの機能を使うために追加

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <random>

#include <vector>
#include <utility>
#include "mcts/BatchQueue.hpp"


#include "board/DynamicKomi.hpp"
#include "board/GoBoard.hpp"
#include "board/Point.hpp"
#include "common/Message.hpp"
#include "pattern/PatternHash.hpp"
#include "feature/Ladder.hpp"
#include "feature/Seki.hpp"
#include "feature/Semeai.hpp"
#include "mcts/MoveSelection.hpp"
#include "mcts/Simulation.hpp"
#include "mcts/UctRating.hpp"
#include "mcts/UctSearch.hpp"
#include "mcts/ucb/UCBEvaluation.hpp"
#include "util/Utility.hpp"
#include "mcts/BatchHandler.hpp" // 先ほど作成したヘッダーをインクルード

static volatile bool interrupted = false;
extern torch::jit::script::Module tamago_model;
torch::Device device(torch::kCPU);

#if defined (_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#endif

/**
 * @brief Policy Mapを1回だけ表示するためのフラグ
 */
static std::atomic<bool> is_first_policy_output(true);

const double C_PUCT = 1.0;
/**
 * @~english
 * @brief UCT nodes.
 * @~japanese
 * @brief UCTノード
 */
uct_node_t *uct_node;

/**
 * @~english
 * @brief Progressive widening threshold for node pruning.
 * @~japanese
 * @brief Progressive Widening の枝刈り探索回数閾値
 */
static int pw[PURE_BOARD_MAX + 1];  

/**
 * @~english
 * @brief Node expansion threshold.
 * @~japanese
 * @brief ノード展開の閾値
 */
static int expand_threshold = EXPAND_THRESHOLD_19;

/**
 * @~english
 * @brief Current root's index.
 * @~japanese
 * @brief 現在のルートのインデックス
 */
int current_root;

/**
 * @~english
 * @brief Mutex variable for UCT nodes.
 * @~japanese
 * @brief UCTノード用のミューテックス変数
 */
static std::mutex *mutex_nodes;

/**
 * @~english
 * @brief Mutex variable for node expansion.
 * @~japanese
 * @brief ノード展開を排他処理するためのミューテックス変数
 */
static std::mutex mutex_expand;       

/**
 * @~english
 * @brief The number of search worker threads.
 * @~japanese
 * @brief 探索スレッド数
 */
static int threads = 1;

/**
 * @~english
 * @brief Arguments for search worker threads.
 * @~japanese
 * @brief 各スレッドに渡す引数
 */
static thread_arg_t t_arg[THREAD_MAX];

/**
 * @~english
 * @brief Statistic information of Monte-Carlo simulation.
 * @~japanese
 * @brief プレイアウトの統計情報
 */
static statistic_t statistic[BOARD_MAX];  

/**
 * @~english
 * @brief Counter for statistic information of Monte-Carlo simulation.
 * @~japanese
 * @brief プレイアウトの統計情報を収集した回数
 */
static std::atomic<int> statistic_count;

/**
 * @~english
 * @brief Criticality value.
 * @~japanese
 * @brief 盤上の各点のCriticality
 */
static double criticality[BOARD_MAX];  

/**
 * @~english
 * @brief Ownership value.
 * @~japanese
 * @brief 盤上の各点のOwner(0-100%)
 */
static double owner[BOARD_MAX];  

/**
 * @~english
 * @brief Indices for ownership feature.
 * @~japanese
 * @brief 現在のオーナーのインデックス
 */
static int owner_index[BOARD_MAX];   

/**
 * @~english
 * @brief Indices for criticality feature.
 * @~japanese
 * @brief 現在のクリティカリティのインデックス
 */
static int criticality_index[BOARD_MAX];  

/**
 * @~english
 * @brief Upper bound value for criticality.
 * @~japanese
 * @brief Criticalityの上限値
 */
static int criticality_max = CRITICALITY_MAX;

/**
 * @~english
 * @brief Move candidate flags.
 * @~japanese
 * @brief 候補手のフラグ
 */
static bool candidates[BOARD_MAX];  

/**
 * @~english
 * @brief Pondering activation flag.
 * @~japanese
 * @brief 予測読みするかどうかのフラグ
 */
bool pondering_mode = false;

/**
 * @~english
 * @brief Pondering flag.
 * @~japanese
 * @brief 予測読み中であることを表すフラグ
 */
static bool ponder = false;

/**
 * @~english
 * @brief Pondering stopping flag.
 * @~japanese
 * @brief 予測読みを止めるためのフラグ
 */
static bool pondering_stop = false;

/**
 * @~english
 * @brief Pondering is executed.
 * @~japanese
 * @brief 予測読みを実行したかどうかを表すフラグ
 */
static bool pondered = false;

/**
 * @~english
 * @brief Search worker thread.
 * @~japanese
 * @brief 探索ワーカスレッド
 */
static std::thread *worker[THREAD_MAX];

/**
 * @~english
 * @brief Random number generator for Monte-Carlo simulation.
 * @~japanese
 * @brief 乱数生成器
 */
static std::mt19937_64 mt[THREAD_MAX];

/**
 * @~english
 * @brief Reuse subtree flag.
 * @~japanese
 * @brief 探索木の再利用のフラグ
 */
static bool reuse_subtree = false;

/**
* @~english
* @brief Ray's stone color.
* @~japanese
* @brief 自分の手番の色
*/
static int my_color;


// Criticaliityの計算
static void CalculateCriticality( int color );

// Criticality
static void CalculateCriticalityIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index );

// Ownershipの計算
static void CalculateOwner( int color );

// Ownership
static void CalculateOwnerIndex( uct_node_t *node, statistic_t *node_statistc, int color, int *index );

// 現局面の子ノードのインデックスの導出
static void CorrectDescendentNodes( std::vector<int> &indexes, int index );

// ノードの展開
static int ExpandNode(game_info_t *game, int color, int current, torch::Tensor policy_tensor);

// ルートの展開
static int ExpandRoot( game_info_t *game, int color );

// UCT探索
static void ParallelUctSearch( thread_arg_t *arg );

// UCT探索(予測読み)
static void ParallelUctSearchPondering( thread_arg_t *arg );

// ノードのレーティング
static void RatingNode( game_info_t *game, int color, int index );

static int RateComp( const void *a, const void *b );

// UCB値が最大の子ノードを返す
static int SelectMaxUcbChild( uct_node_t &node, const int moves, const int color, std::mt19937_64 &mt );

// 各座標の統計処理
static void Statistic( game_info_t *game, int winner );

// UCT探索(1回の呼び出しにつき, 1回の探索)
static double UctSearch(game_info_t *game, int color, std::mt19937_64 &mt, int current, std::vector<std::pair<int, int>> &path);

// ノード展開の閾値を取得
static int GetExpandThreshold( const game_info_t *game );

BatchQueue g_batch_queue; 
extern double const_playout;
extern int threads;

// 2. BatchHandler.cpp にある関数の宣言
extern void ProcessMiniBatch(torch::jit::script::Module &model, torch::Device &device);

extern int GetPlayoutLimit(void);             // po_limit取得用
extern void IncrementPoCount(void);           // Atomicの代わり
extern void PrintSearchStatus( int root_index );
extern void UpdateNodeStats( int current, int child_index, double v );

int UctSearchGenmove(struct game_info_t *game, int color, int mode);

/**
 * @~english
 * @brief Get node's data.
 * @param[in] index Index for node.
 * @return Node's data.
 * @~japanese
 * @brief 指定したインデックスのノードのデータの取得
 * @param[in] index ノードのインデックス
 * @return 指定したインデックスのノードのデータ
 */
uct_node_t&
GetNode( const int index )
{
  return uct_node[index];
}


/**
 * @~english
 * @brief Get current root node's data.
 * @return Current root node's data.
 * @~japanese
 * @brief 現在のルートノードのデータの取得
 * @return 現在のルートノードのデータ
 */
uct_node_t&
GetRootNode( void )
{
  return uct_node[current_root];
}


/**
 * @~english
 * @brief Set pondering mode.
 * @param[in] flag Pondering mode.
 * @~japanese
 * @brief 予測読みの設定
 * @param[in] flag 予測読みのモード
 */
void
SetPonderingMode( const bool flag )
{
  pondering_mode = flag;
}


/**
 * @~english
 * @brief Set the number of search worker threads.
 * @param[in] new_threads The number of search worker threads.
 * @~japanese
 * @brief 使用するスレッド数の指定
 * @param[in] new_threads 使用するスレッド数
 */
void
SetThread( const int new_threads )
{
  threads = new_threads;
}


/**
 * @~english
 * @brief Set reuse subtre mode.
 * @param[in] flag Reuse subtree mode.
 * @~japanese
 * @brief 探索木再利用の設定
 * @param[in] flag 探索木再利用のモード
 */
void
SetReuseSubtree( bool flag )
{
  reuse_subtree = flag;
}


/**
 * @~english
 * @brief Set parameters for search settings.
 * @~japanese
 * @brief 探索用パラメータの設定
 */
void
SetParameter( void )
{
  if (pure_board_size < 11) {
    expand_threshold = EXPAND_THRESHOLD_9;
  } else if (pure_board_size < 16) {
    expand_threshold = EXPAND_THRESHOLD_13;
  } else {
    expand_threshold = EXPAND_THRESHOLD_19;
  }

  SetTimeManagementParameter();
}


/**
 * @~english
 * @brief Initialization for tree search.
 * @~japanese
 * @brief 木探索の初期設定
 */
void
InitializeUctSearch( void )
{
  int i;

  // Progressive Wideningの初期化  
  pw[0] = 0;
  for (i = 1; i <= PURE_BOARD_MAX; i++) {  
    pw[i] = pw[i - 1] + static_cast<int>(40 * pow(PROGRESSIVE_WIDENING, i - 1));
    if (pw[i] > 10000000) break;
  }
  for (i = i + 1; i <= PURE_BOARD_MAX; i++) { 
    pw[i] = INT_MAX;
  }

  mutex_nodes = new std::mutex[uct_hash_size];

  // UCTのノードのメモリを確保
  uct_node = new uct_node_t[uct_hash_size];

  std::cerr << "Require " << uct_hash_size * sizeof(uct_node_t) / 1024 / 1024 << " Mbytes for Uct Node" << std::endl << std::endl;
  std::cerr << sizeof(uct_node_t) << std::endl;
  std::cerr << sizeof(child_node_t) * UCT_CHILD_MAX << std::endl;
  
  if (uct_node == NULL) {
    std::cerr << "Cannot allocate memory !!" << std::endl;
    std::cerr << "You must reduce tree size !!" << std::endl;
    exit(1);
  }

}


/**
 * @~english
 * @brief Initialization for search settings.
 * @~japanese
 * @brief 探索設定の初期化
 */
void
InitializeSearchSetting( void )
{
  // Ownerの初期化
  for (int i = 0; i < board_max; i++){
    owner[i] = 50;
    owner_index[i] = 5;
    candidates[i] = true;
  }

  // 乱数の初期化
  std::random_device rand;
  for (int i = 0; i < THREAD_MAX; i++) {
    mt[i].seed(rand());
  }


  SetTimeManagementParameter();
  InitializeTimeSetting();

  pondered = false;
  pondering_stop = true;
}


/**
 * @~english
 * @brief Stop pondering.
 * @~japanese
 * @brief 予測読みの停止
 */
void
StopPondering( void )
{
  if (!pondering_mode) {
    return ;
  }

  if (ponder) {
    pondering_stop = true;
    for (int i = 0; i < threads; i++) {
      worker[i]->join();
      delete worker[i];
    }
    ponder = false;
    pondered = true;
    PrintPonderingCount(GetPoCount());
  }
}

extern void InitializeUctSearch(void);

/**
 * @brief 思考エンジン本体（メインループ）
 * @param[in] game 現在の局面
 * @param[in] color コンピュータの手番の色
 * @return 選択された着手（pos）
 */
int
UctSearchGenmove(game_info_t *game, int color, int mode)
{
    // --- 1. ルートノード(0番)およびツリーのリセット ---
    // メモリ再確保を避けるため、InitializeUctSearch()は呼ばず最小限のリセットを行う
    int root = 0;
    uct_node[root].child_num = 0; 
    uct_node[root].move_count = 0;

    // Ray内部定数 pure_board_max を使用（環境により board_max 等へ変更）
    for (int j = 0; j < pure_board_max; j++) {
        uct_node[root].child[j].index = NOT_EXPANDED;
        uct_node[root].child[j].move_count = 0;
        uct_node[root].child[j].win = 0.0;
        uct_node[root].child[j].virtual_loss = 0;
    }

    // ハッシュテーブルのクリア（盤面とノードの紐付けをリセット）
    ClearUctHash();

    // 探索設定
    int po_limit = 1000; 
    int thread_num = static_cast<int>(threads); 
    if (thread_num <= 0) thread_num = 1;

    ResetPoCount();
    interrupted = false;

    // --- 2. ルート展開 (初回の評価) ---
    // ルートが未展開の場合、スレッド起動前にNNで評価しておく
    if (uct_node[root].child_num == 0) {
        at::Tensor input = TamaGoFeature::GenerateInputPlanes(game, color);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input.to(device));
        auto outputs = tamago_model.forward(inputs).toTuple();
        auto policy = outputs->elements()[0].toTensor().softmax(1).to(torch::kCPU).view({-1});
        ExpandNode(game, color, root, policy);
        fprintf(stderr, "DEBUG: Root Expanded. Moves: %d\n", uct_node[root].child_num);
    }

    // --- 3. ワーカースレッドの起動 ---
    std::vector<std::thread> workers;
    for (int i = 0; i < thread_num; i++) {
        workers.emplace_back([&, i]() {
            fprintf(stderr, "DEBUG: [Worker %d] Started.\n", i);
            std::mt19937_64 mt(std::random_device{}());
            
            while (GetPoCount() < po_limit && !interrupted) {
                game_info_t *copy_game = AllocateGame();
                CopyGame(copy_game, game);
                
                std::vector<std::pair<int, int>> path;
                
                // 探索実行（内部でキューに積まれる場合は path に記録される想定）
                UctSearch(copy_game, color, mt, root, path);
                
                // UctSearch 内で copy_game を FreeGame しない実装の場合、ここで解放
                FreeGame(copy_game);
                IncrementPoCount();
            }
            fprintf(stderr, "DEBUG: [Worker %d] Finished.\n", i);
        });
    }

    // --- 4. メインスレッドによるミニバッチ推論ループ ---
    fprintf(stderr, "DEBUG: [Main] Loop Start. Target PO: %d\n", po_limit);

    // GetPoCount() だけでなく、全スレッドがリクエストを出し切るまで回す
    while (GetPoCount() < po_limit && !interrupted) {
        if (g_batch_queue.size() > 0) {
            // 推論実行
            ProcessMiniBatch(tamago_model, device);
            // ログの頻度を抑える
            static int batch_tick = 0;
            if (++batch_tick % 10 == 0) {
                fprintf(stderr, "DEBUG: [Main] Processed Batch. Current PO: %d\n", GetPoCount());
            }
        } else {
            // リクエストがない間は CPU を明け渡す
            std::this_thread::yield();
        }
    }
    fprintf(stderr, "DEBUG: [Main] Loop Finished.\n");

    // --- 5. 終了処理 ---
    // interrupted を true にしてワーカースレッドに終了を促す
    interrupted = true; 
    for (auto &t : workers) {
        if (t.joinable()) t.join();
    }

    // 残っている推論リクエストを全て処理
    while (g_batch_queue.size() > 0) {
        ProcessMiniBatch(tamago_model, device);
    }

    // --- 6. 結果の集計と着手選択 ---
    double best_wp = 0.0;
    
    fprintf(stderr, "--- Root Node Stats ---\n");
    for (int i = 0; i < uct_node[root].child_num; ++i) {
        auto& c = uct_node[root].child[i];
        if (c.move_count > 0) {
            double wp = (double)c.win / c.move_count;
            fprintf(stderr, "pos=%3d n=%4d win=%7.2f wp=%.4f\n", 
                    c.pos, (int)c.move_count, (double)c.win, wp);
        }
    }

    // SelectMove は Ray の標準的なもの、あるいは自作のものを呼び出し
    int select_pos = SelectMove(game, uct_node[root], color, best_wp);

    fprintf(stderr, "-----------------------\n");
    fprintf(stderr, "SELECTED MOVE: %d with WP: %.4f\n", select_pos, best_wp);

    return select_pos;
}

void UpdateNodeStats(int current, int child_index, double v) {
    // 子ノードの統計更新
    uct_node[current].child[child_index].move_count.fetch_add(1);
    
    // win (double) のAtomic加算
    auto& atomic_child_win = reinterpret_cast<std::atomic<double>&>(uct_node[current].child[child_index].win);
    double old_child_win = atomic_child_win.load();
    while (!atomic_child_win.compare_exchange_weak(old_child_win, old_child_win + v));

    // 親ノード（自分）の統計更新
    uct_node[current].move_count.fetch_add(1);
    
    // 親ノードの win には 1.0 - v を加算する場合が多い（視点の反転）
    auto& atomic_node_win = reinterpret_cast<std::atomic<double>&>(uct_node[current].win);
    double old_node_win = atomic_node_win.load();
    while (!atomic_node_win.compare_exchange_weak(old_node_win, old_node_win + (1.0 - v)));
}

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <vector>

// NNのPolicy(0-81)をRayの座標(1次元インデックス)に変換するテーブル
// ※初期化時に一度だけ作成するか、ループ内で計算します。
// NNの0-80をRayの内部座標(int)に変換する
static int NNIdxToRayPos(int nn_idx, int bsize) {
    if (nn_idx == bsize * bsize) return PASS;

    char tmp_pos[10];
    // x方向: 'I' を飛ばす囲碁の慣習に対応したテーブル (gogui_x は 'A','B',...,'H','J',...)
    // nn_idx % 9 = 0 -> 'A', 1 -> 'B', ...
    char x_char = gogui_x[(nn_idx % bsize) + 1]; 
    
    // y方向: NNの 0-8 行目を 1-9 行目に変換 (上が 9, 下が 1)
    // NNの nn_idx / 9 = 0 が「一番上の行(9)」を指している場合
    int y_num = bsize - (nn_idx / bsize);
    
    sprintf(tmp_pos, "%c%d", x_char, y_num);
    
    // Ray 本来のロジックで数値座標に変換
    return StringToInteger(tmp_pos);
}

enum UCT_RESULT {
    UCT_RESULT_LOSS = 0,
    UCT_RESULT_WIN = 1,
};

/**
 * @~english
 * @brief Ponder by UCT search.
 * @param[in] game Current board position data.
 * @param[in] color Player's color.
 * @param[in] lz_analysis_cs Update interval for lz-analyze.
 * @return Coordinate of next move.
 * @~japanese
 * @brief UCT探索による予測読み
 * @param[in] game 現在の局面情報
 * @param[in] color 手番の色
 * @param[in] lz_analysis_cs lz-analyzeコマンドの更新間隔
 * @return 次の着手の座標
 */
void
UctSearchPondering( game_info_t *game, int color, int lz_analysis_cs )
{
  if (!pondering_mode) {
    return;
  }

  // 探索情報をクリア
  for (int i = 0; i < board_max; i++) {
    statistic[i].clear();
  }
  statistic_count = 0;
  std::fill_n(criticality_index, board_max, 0);  
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;    
  }

  ResetPoCount();

  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    owner[pos] = 50;
    owner_index[pos] = 5;
    candidates[pos] = true;
  }

  // UCTの初期化
  current_root = ExpandRoot(game, color);

  pondered = false;

  // 子ノードが1つ(パスのみ)ならPASSを返す
  if (uct_node[current_root].child_num <= 1) {
    ponder = false;
    pondering_stop = true;
    return;
  }

  ponder = true;
  pondering_stop = false;

  // Dynamic Komiの算出(置碁のときのみ)
  DynamicKomi(game, &uct_node[current_root], color);

  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    t_arg[i].lz_analysis_cs = lz_analysis_cs;
    worker[i] = new std::thread(ParallelUctSearchPondering, &t_arg[i]);
  }

  return;
}

/**
 * @brief ルートノードを展開し、ニューラルネットワーク(TamaGo)のポリシーで初期化する
 */
/**
 * @brief ルートノードを展開し、ニューラルネットワーク(TamaGo)のポリシーで初期化する
 */
static int
ExpandRoot(game_info_t *game, int color)
{
  const int moves = game->moves;
  const unsigned long long hash = game->move_hash;
  unsigned int index = FindSameHashIndex(hash, color, moves);
  int pm1 = PASS, pm2 = PASS;
  bool ladder[BOARD_MAX] = { false };

  // 直前の着手の座標を取り出す
  pm1 = game->record[moves - 1].pos;
  if (moves > 1) pm2 = game->record[moves - 2].pos;

  if (pure_board_size != 9) {
    LadderExtension(game, color, ladder);
  }

  // --- A. 既存ノードの再利用 ---
  if (index != uct_hash_size) {
    std::vector<int> indexes;
    CorrectDescendentNodes(indexes, index);
    std::sort(indexes.begin(), indexes.end());
    ClearNotDescendentNodes(indexes);
    
    uct_node[index].previous_move1 = pm1;
    uct_node[index].previous_move2 = pm2;

    ReuseRootCandidateWithoutLadderMove(uct_node[index], ladder);
    uct_node[index].width = 1;

    PrintReuseCount(uct_node[index].move_count);

    return (int)index;
  } 
  // --- B. 新規ノードの作成 ---
  else {
    ClearUctHash();
    index = SearchEmptyIndex(hash, color, moves);
    if (index == uct_hash_size) return -1; 
    
    InitializeNode(uct_node[index], pm1, pm2);

    // --- ニューラルネットワークによる推論 ---
    // 1. 入力特徴量の作成（TamaGoFeatureを使用）
    at::Tensor input = TamaGoFeature::GenerateInputPlanes(game, color);
    torch::NoGradGuard no_grad;

    // 2. デバイスの決定と転送
    // イテレータをデリファレンスしてデバイスを取得
    auto params = tamago_model.parameters();
    if (params.begin() != params.end()) {
        input = input.to((*params.begin()).device());
    }

    // 3. 推論実行
    auto outputs = tamago_model.forward({input}).toTuple();
    
    // 【Segmentation fault 対策】
    // 戻り値のテンソル [1, 82] を view({-1}) で [82] に平坦化します。
    // これを行わないと accessor<float, 1> が次元不一致でクラッシュします。
    at::Tensor policy_tensor = outputs->elements()[0].toTensor().softmax(1).to(torch::kCPU).view({-1});
    auto p_acc = policy_tensor.accessor<float, 1>();

    child_node_t *uct_child = uct_node[index].child;
    int child_num = 0;

    // 4. パスノードの展開 (TamaGoの出力インデックス 81)
    if (child_num < UCT_CHILD_MAX) {
        InitializeCandidate(uct_child[child_num], child_num, PASS, ladder[PASS]);
        // pure_board_max は 9x9 なら 81
        uct_child[child_num].nn_policy = (float)p_acc[pure_board_max]; 
        child_num++; 
    }
    
    // 5. 盤面上の候補手の展開
    for (int i = 0; i < pure_board_max; i++) {
        const int pos = onboard_pos[i];
        
        // 合法手チェック
        if (IsLegal(game, pos, color) && IsMeaningfulSelfAtari(game, color, pos)) {
            if (child_num >= UCT_CHILD_MAX) break; 
            
            InitializeCandidate(uct_child[child_num], child_num, pos, ladder[pos]);
            // ニューラルネットワークのポリシーを反映
            uct_child[child_num].nn_policy = (float)p_acc[i];
            child_num++;
        }
    }
    
    uct_node[index].child_num = child_num;
    CheckSeki(game, uct_node[index].seki);
    uct_node[index].width++;
  }

  return (int)index;
}


/**
 * @brief ノードの展開 (TamaGo Policy反映版)
 * @param[in] game 現在の局面情報
 * @param[in] color 手番の色
 * @param[in] current 現在のノードのインデックス
 * @param[in] policy_tensor TamaGoの推論結果（Policyヘッドの出力）
 * @return 展開されたノードのインデックス
 */
/**
 * @brief NNの推論結果（Policy）を用いて、新しいUCTノードを展開する
 */
static int
ExpandNode(game_info_t *game, int color, int current, torch::Tensor policy_tensor)
{
    const int moves = game->moves;
    const unsigned long long hash = game->move_hash;
    unsigned int index = FindSameHashIndex(hash, color, moves);
    int pm1 = PASS, pm2 = PASS;
    
    if (index != uct_hash_size) return (int)index;

    index = SearchEmptyIndex(hash, color, moves);
    if (index == uct_hash_size) return -1;

    if (moves > 0) pm1 = game->record[moves - 1].pos;
    if (moves > 1) pm2 = game->record[moves - 2].pos;

    InitializeNode(uct_node[index], pm1, pm2);

    child_node_t *uct_child = uct_node[index].child;
    int child_num = 0;

    // --- NN Policyデータの取得 ---
    auto policy_cpu = policy_tensor.to(torch::kCPU);
    float* p_ptr = policy_cpu.data_ptr<float>();

    // 3. 盤上の候補手の展開 (0-80)
    for (int i = 0; i < 81; i++) {
        // TamaGo index (0-80) -> x, y (1-9)
        // TamaGoは左上(0,0)をインデックス0とする
        int x = (i % 9) + 1;
        int y = (i / 9) + 1;

        // Ray内部座標系への変換
        // OB_SIZEを含めた正しいオフセット計算
        int ray_pos = POS(x + OB_SIZE - 1, y + OB_SIZE - 1);

        if (i == 0) { // ループの最初だけでOK
            int stones = 0;
            for (int p = 0; p < BOARD_MAX; p++) {
                if (game->board[p] == S_BLACK || game->board[p] == S_WHITE) stones++;
            }
            std::cerr << "DEBUG: ExpandNode - Current board stones: " << stones << std::endl;
        }

        if (IsLegal(game, ray_pos, color)) {
            InitializeCandidate(uct_child[child_num], child_num, ray_pos, false);
            uct_child[child_num].nn_policy = p_ptr[i]; 
        } 
        /* // もしこれでも直らない場合、以下のコメントを外してビルドしてください
        else if (p_ptr[i] > 0.01) {
            std::cerr << "Illegal Move: i=" << i << " x=" << x << " y=" << y << " pos=" << ray_pos << std::endl;
        }
        */
    }

    // 4. パスの展開 (TamaGo index 81)
    if (IsLegal(game, PASS, color)) {
        InitializeCandidate(uct_child[child_num], child_num, PASS, false);
        uct_child[child_num].nn_policy = p_ptr[81]; 
    }

    uct_node[index].child_num = child_num;
    CheckSeki(game, uct_node[index].seki);
    uct_node[index].width = 1;

    // 6. 知識継承
    if (current >= 0 && current < (int)uct_hash_size) {
        const int sibling_num = uct_node[current].child_num;
        child_node_t *uct_sibling = uct_node[current].child;
        double max_rate = -1.0;
        int max_pos = PASS;

        for (int i = 0; i < sibling_num; i++) {
            if (uct_sibling[i].pos != pm1) {
                if (uct_sibling[i].rate > max_rate) {
                    max_rate = uct_sibling[i].rate;
                    max_pos = uct_sibling[i].pos;
                }
            }
        }

        if (max_pos != PASS) {
            for (int i = 0; i < uct_node[index].child_num; i++) {
                if (uct_child[i].pos == max_pos) {
                    if (!uct_child[i].pw) uct_child[i].open = true;
                    break;
                }
            }
        }
    }

    return (int)index;
}

/**
 * @~english
 * @brief Calculate all candidates' move score.
 * @param[in] game Board position data.
 * @param[in] color Player's color.
 * @param[in] index Node index.
 * @~japanese
 * @brief 着手のスコアの計算
 * @param[in] game 局面情報
 * @param[in] color 手番の色
 * @param[in] index ノードのインデックス
 */
static void
RatingNode( game_info_t *game, int color, int index )
{
  const int child_num = uct_node[index].child_num;
  const int moves = game->moves;
  int max_index;
  double score = 0.0, max_score, dynamic_parameter, total_score = 0.0;
  child_node_t *uct_child = uct_node[index].child;
  unsigned int tactical_features[BOARD_MAX * UCT_INDEX_MAX] = {0};
  int distance_index = 0;

  // 【追加】TamaGoによる盤面全体の推論（1回実行）
  torch::NoGradGuard no_grad;
  torch::Tensor input = TamaGoFeature::Extract(game, color);
  auto outputs = tamago_model.forward({input}).toTuple()->elements();
  torch::Tensor policy = torch::softmax(outputs[0].toTensor(), 1);

  // --- Valueの取得と計算 ---
  // outputs[1] は [1, 3] の形状のテンソル
  torch::Tensor value_logits = outputs[1].toTensor();
  // Softmaxをかけて確率 (0.0~1.0) に変換
  torch::Tensor value_probs = torch::softmax(value_logits, 1); 

  // 値を取り出す (平坦化してアクセス)
  float p_loss = value_probs[0][0].item<float>(); // 負けの確率
  float p_draw = value_probs[0][1].item<float>(); // 引き分けの確率
  float p_win  = value_probs[0][2].item<float>(); // 勝ちの確率

  // 期待値としての勝率を計算 (0.0 ~ 1.0)
  // 勝ちを1.0、引き分けを0.5、負けを0.0として加重平均
  double win_rate = (p_win * 1.0) + (p_draw * 0.5);

  // デバッグ表示
  std::cerr << "Value Probabilities -> Loss: " << (p_loss*100) 
            << "% | Draw: " << (p_draw*100) 
            << "% | Win: " << (p_win*100) << "%" << std::endl;
  std::cerr << "Expected Win Rate: " << (win_rate * 100.0) << "%" << std::endl;

  // パスのレーティング
  uct_child[PASS_INDEX].rate = CalculateMoveScoreWithBTFM(game, PASS, tactical_features, distance_index);
  // 直前の着手で発生した特徴の確認
  distance_index = CheckFeaturesForTree(game, color, tactical_features);
  // 直前の着手で石を2つ取られたか確認
  CheckRemove2StonesForTree(game, color, tactical_features);
  // 2手前で劫が発生していたら, 劫を解消するトリの確認
  if (game->ko_move == moves - 2) {
    CheckCaptureAfterKoForTree(game, color, tactical_features);
    CheckKoConnectionForTree(game, tactical_features);
  } else if (game->ko_move == moves - 3) {
    CheckKoRecaptureForTree(game, color, tactical_features);
  }

  max_index = 0;
  max_score = uct_child[0].rate;
  for (int i = 0; i < child_num; i++) {
    const int pos = uct_child[i].pos;

    if (pos == PASS) {
      // --- パスの場合：TamaGoモデルの81番目の値を取得 ---
      // TamaGoの仕様: 9x9=81番目がPASS
      score = policy[0][81].item<double>(); 
    } else {
      // --- 着手の場合：座標からインデックスを計算 ---
      int x = X(pos) - 1; // 1~9 -> 0~8
      int y = Y(pos) - 1; // 1~9 -> 0~8
      
      // 安全策：インデックスが範囲内(0-80)かチェック
      int idx = y * pure_board_size + x;
      if (idx >= 0 && idx < 81) {
        score = policy[0][idx].item<double>();
      } else {
        score = 0.0; // 異常な座標の場合はスコア0
      }

      // --- B. 元のコードにある戦術チェックを実行（副作用を利用） ---
      CheckSelfAtariForTree(game, color, pos, tactical_features);
      CheckCaptureForTree(game, color, pos, tactical_features);
      CheckAtariForTree(game, color, pos, tactical_features);

      // --- C. シチョウなどの致命的なルール違反にペナルティ ---
      if (uct_child[i].ladder) {
        score *= 0.0001; // シチョウで死ぬ手はスコアを極小に
      }
      
      // 元のコードで score = 0.0; としていたような「悪手」の判定があれば
      // ここで score をさらに調整できます。
    }

    uct_child[i].rate = score;
    total_score += score;

    // Ray独自の統計補正（Owner/Criticality）はそのまま適用
    double dynamic_parameter = uct_owner[owner_index[pos]] * uct_criticality[criticality_index[pos]];

    if (i == 0 || score * dynamic_parameter > max_score) {
      max_index = i;
      max_score = score * dynamic_parameter;
    }
  }

  // 正規化処理
  const double inv_total = 1.0 / total_score;
  for (int i = 0; i < child_num; i++) {
    uct_child[i].rate *= inv_total;
  }
  
  uct_child[max_index].pw = true;
}
/*
  for (int i = 1; i < child_num; i++) {
    const int pos = uct_child[i].pos;
  
    // 自己アタリの確認
    CheckSelfAtariForTree(game, color, pos, tactical_features);
    // トリの確認
    CheckCaptureForTree(game, color, pos, tactical_features);
    // アタリの確認
    CheckAtariForTree(game, color, pos, tactical_features);

    // 自己アタリが無意味だったらスコアを0.0にする
    // 逃げられないシチョウならスコアを-1.0にする
    if (uct_child[i].ladder) {
      score = 0.0;
    } else {
      score = CalculateMoveScoreWithBTFM(game, pos, tactical_features, distance_index);
    }

    // その手のγを記録
    uct_child[i].rate = score;
    total_score += score;

    // 現在見ている箇所のOwnerとCriticalityの補正値を求める
    //dynamic_parameter = 1.0;
    dynamic_parameter = uct_owner[owner_index[pos]] * uct_criticality[criticality_index[pos]];

    // 最もγが大きい着手を記録する
    if (score * dynamic_parameter > max_score) {
      max_index = i;
      max_score = score * dynamic_parameter;
    }
  }

  const double inv_total = 1.0 / total_score;

  for (int i = 0; i < child_num; i++) {
    uct_child[i].rate *= inv_total;
  }
  
  // 最もγが大きい着手を探索できるようにする
  uct_child[max_index].pw = true;
}
  */


/**
 * @~english
 * @brief Search worker.
 * @param[in] arg Arguments for a search worker thread.
 * @~japanese
 * @brief 探索ワーカ
 * @param[in] arg 探索ワーカスレッドの引数
 */
static void
ParallelUctSearch( thread_arg_t *arg )
{
  const thread_arg_t *targ = (thread_arg_t *)arg;
  const int color = targ->color;
  bool interruption = false, enough_size = true;
  bool use_analysis = targ->lz_analysis_cs > 0 ? true : false;
  int winner = 0, interval = CRITICALITY_INTERVAL;
  std::vector<std::pair<int, int>> path; // pathを用意
  game_info_t *game = AllocateGame();
  ray_clock::time_point analysis_timer;

  // --- 外部フラグを参照 ---
  // UctSearch.cpp 上部で定義した static volatile bool interrupted; を使用
  extern volatile bool interrupted; 

  // スレッドIDが0のスレッドだけ別の処理をする
  // 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
  if (targ->thread_id == 0) {
    analysis_timer = ray_clock::now();

    do {
      // 探索回数を1回増やす
      IncrementPoCount();
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
      std::vector<std::pair<int, int>> path; // 探索ごとに経路保存用の容器を作る
      UctSearch(game, color, mt[targ->thread_id], current_root, path);
      // 探索を打ち切るか確認
      interruption = CheckInterruption(uct_node[current_root]);
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();
      // OwnerとCriticalityを計算する
      if (GetPoCount() > interval) {
        CalculateOwner(color);
        CalculateCriticality(color);
        interval += CRITICALITY_INTERVAL;
      }

      if (use_analysis &&
          static_cast<int>(100 * GetSpendTime(analysis_timer)) > targ->lz_analysis_cs) {
        analysis_timer = ray_clock::now();
        PrintLeelaZeroAnalyze(&uct_node[current_root]);
      }

      // 時間切れ、またはメインスレッドからの停止命令を確認
      if (IsTimeOver() || interrupted) break;

    } while (GetPoCount() < GetPoHalt() && !interruption && enough_size && !interrupted);
  } else {
    do {
      // 探索回数を1回増やす
      IncrementPoCount();
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
      std::vector<std::pair<int, int>> path; // 探索ごとに経路保存用の容器を作る
      UctSearch(game, color, mt[targ->thread_id], current_root, path);
      // 探索を打ち切るか確認
      interruption = CheckInterruption(uct_node[current_root]);
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();

      // 時間切れ、またはメインスレッドからの停止命令を確認
      if (IsTimeOver() || interrupted) break;

    } while (GetPoCount() < GetPoHalt() && !interruption && enough_size && !interrupted);
  }

  // メモリの解放
  FreeGame(game);
}


/**
 * @~english
 * @brief Pondering worker.
 * @param[in] arg Arguments for a search worker thread.
 * @~japanese
 * @brief 予測読みワーカ
 * @param[in] arg 予測読みワーカスレッドの引数
 */
static void
ParallelUctSearchPondering( thread_arg_t *arg )
{
  const thread_arg_t *targ = (thread_arg_t *)arg;
  const int color = targ->color;
  int winner = 0, interval = CRITICALITY_INTERVAL;
  bool enough_size = true;
  bool use_analysis = targ->lz_analysis_cs > 0 ? true : false;
  game_info_t *game = AllocateGame();
  ray_clock::time_point analysis_timer;

  // スレッドIDが0のスレッドだけ別の処理をする
  // 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
  if (targ->thread_id == 0) {
    analysis_timer = ray_clock::now();
    do {
      // 探索回数を1回増やす
      IncrementPoCount();
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
      std::vector<std::pair<int, int>> path; // 探索ごとに経路保存用の容器を作る
      UctSearch(game, color, mt[targ->thread_id], current_root, path);
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();
      // OwnerとCriticalityを計算する
      if (GetPoCount() > interval) {
        CalculateOwner(color);
        CalculateCriticality(color);
        interval += CRITICALITY_INTERVAL;
      }

      if (use_analysis &&
          static_cast<int>(100 * GetSpendTime(analysis_timer)) > targ->lz_analysis_cs) {
        analysis_timer = ray_clock::now();
        PrintLeelaZeroAnalyze(&uct_node[current_root]);
      }

    } while (!pondering_stop && enough_size);
  } else {
    do {
      // 探索回数を1回増やす
      IncrementPoCount();
      // 盤面のコピー
      CopyGame(game, targ->game);
      // 1回プレイアウトする
      std::vector<std::pair<int, int>> path; // 探索ごとに経路保存用の容器を作る
      UctSearch(game, color, mt[targ->thread_id], current_root, path);
      // ハッシュに余裕があるか確認
      enough_size = CheckRemainingHashSize();
    } while (!pondering_stop && enough_size);
  }

  // メモリの解放
  FreeGame(game);
}

/**
 * @brief モンテカルロ木探索の再帰実行
 * @param[in] game 現在の局面情報
 * @param[in] color 現在の手番の色
 * @param[in] mt 乱数生成器
 * @param[in] current 現在のノードインデックス
 * @param[in,out] path 辿ったノードと選択した子のインデックスを記録する配列
 * @return 探索結果の勝率
 */
static double
UctSearch(game_info_t *game, int color, std::mt19937_64 &mt, int current, std::vector<std::pair<int, int>> &path)
{
    // 1. 終局判定
    bool is_game_over = (game->moves > 1 && 
                         game->record[game->moves - 1].pos == PASS && 
                         game->record[game->moves - 2].pos == PASS);

    if (game->moves >= MAX_MOVES || is_game_over) {
        double score = static_cast<double>(CalculateScore(game));
        extern double komi[]; 
        score -= komi[0]; 

        if (color == S_BLACK) return (score > 0) ? 1.0 : 0.0;
        else return (score < 0) ? 1.0 : 0.0;
    }

    // 2. 次の手を選択（現在のツリー状態からUCBが最大の手を選ぶ）
    int next_index = SelectMaxUcbChild(uct_node[current], game->moves, color, mt);
    int pos = uct_node[current].child[next_index].pos;
    int next_node = uct_node[current].child[next_index].index;

    path.push_back({current, next_index});

    // 3. 次の局面の作成（元の game を汚さないよう常にコピー）
    game_info_t *next_game = AllocateGame();
    CopyGame(next_game, game);
    if (IsLegal(next_game, pos, color)) {
        PutStone(next_game, pos, color);
    }

    double v = 0.5;

    // --- 【検証用】深さ1で止めるロジック ---
    if (next_node == NOT_EXPANDED) {
        // 未展開：NN評価へ回す
        uct_node[current].child[next_index].virtual_loss.fetch_add(1);

        // NN入力用テンソル生成
        at::Tensor input_tensor = TamaGoFeature::GenerateInputPlanes(next_game, 3 - color);
        
        // キューに push。これ以降、next_game の所有権は BatchThread(ProcessMiniBatch) に移る。
        // ※ここでは FreeGame(next_game) は絶対に呼ばない。
        g_batch_queue.push(input_tensor, next_game, path, 3 - color);
        
        v = 0.5; // BatchThread が値を更新するまでの暫定値
    } 
    else {
        // 展開済み：深さ1検証中は再帰せず、既存の統計情報を利用してバックプロパゲーションのテスト
        if (uct_node[next_node].move_count > 0) {
            v = 1.0 - (double)uct_node[next_node].win / uct_node[next_node].move_count;
        } else {
            v = 0.5;
        }

        // 展開済みの場合は BatchThread に行かないため、ここで確実に解放
        FreeGame(next_game);

        // 統計情報の更新
        UpdateNodeStats(current, next_index, v);
    }

    return v;
}
/**
 * @~english
 * @brief Comparator by move evaluation value.
 * @param[in] a Left-hand value.
 * @param[in] b Right-hand value.
 * @return Order judgment.
 * @~japanese
 * @brief 着手評価の大小比較
 * @param[in] a 左辺値
 * @param[in] b 右辺値
 * @return 並び順の判定
 */
static int
RateComp( const void *a, const void *b )
{
  rate_order_t *ro1 = (rate_order_t *)a;
  rate_order_t *ro2 = (rate_order_t *)b;
  if (ro1->rate < ro2->rate) {
    return 1;
  } else if (ro1->rate > ro2->rate) {
    return -1;
  } else {
    return 0;
  }
}


/**
 * @~english
 * @brief Select next move.
 * @param[in] current MCTS node index.
 * @param[in] color Player's color.
 * @param[in] mt Random number generator.
 * @return Node index for next move.
 * @~japanese
 * @brief 次の着手の選択
 * @param[in] current MCTSノードインデックス
 * @param[in] color 手番の色
 * @param[in] mt 乱数生成器
 * @return 次の手に対応するノードのインデックス
 */
static int
SelectMaxUcbChild( uct_node_t &node, const int moves, const int color, std::mt19937_64 &mt )
{
    int select_index = -1;
    double max_score = -1e20;
    
    // UCB/PUCT計算用の定数
    const int total_visits = node.move_count.load();
    const double sqrt_total = std::sqrt(static_cast<double>(total_visits) + 1.0);
    const double C_PUCT = 1.0; // 探索定数（必要に応じて調整）

    // デバッグフラグ：ルートノードの最初の数回の探索を詳細に出す
    static int debug_print_count = 0;
    bool do_debug = (total_visits < 20); // 合計訪問数が少ない（探索初期）時だけ出力

    for (int i = 0; i < node.child_num; i++) {
        child_node_t &child = node.child[i];
        const int n = child.move_count.load() + child.virtual_loss.load();
        
        // 1. Q値 (実績勝率): バックプロパゲーションの結果がここに現れる
        double q_value = 0.5; // 未訪問時の初期値
        if (n > 0) {
            q_value = static_cast<double>(child.win) / n;
        }

        // 2. U値 (Policy項): NNの出力(nn_policy)がここに現れる
        // PUCT公式: C * P * sqrt(Total) / (1 + n)
        double u_value = C_PUCT * child.nn_policy * sqrt_total / (1.0 + n);
        
        double score = q_value + u_value;

        // 【デバッグログ】
        if (do_debug && child.nn_policy > 0.001) {
            char pos_str[10];
            IntegerToString(child.pos, pos_str);
            fprintf(stderr, "  DEBUG: pos=%-3s n=%2d q=%.4f p=%.4f score=%.4f\n",
                    pos_str, n, q_value, child.nn_policy, score);
        }

        if (score > max_score) {
            max_score = score;
            select_index = i;
        } else if (score == max_score) {
            // スコアが同じ場合は乱数でタイブレーク
            if (mt() % 2 == 0) select_index = i;
        }
    }

    if (do_debug && select_index != -1) {
        char best_pos[10];
        IntegerToString(node.child[select_index].pos, best_pos);
        fprintf(stderr, "  >> SELECTED: %s (TotalVisits: %d)\n", best_pos, total_visits);
    }

    return select_index;
}


/**
 * @~english
 * @brief Update statistic information.
 * @param[in] game Board position data.
 * @param[in] winner Winner's color.
 * @~japanese
 * @brief 統計情報の更新
 * @param[in] game 局面情報
 * @param[in] winner 勝った手番の色
 */
static void
Statistic( game_info_t *game, int winner )
{
  const char *board = game->board;

  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    int color = board[pos];

    if (color == S_EMPTY) {
      color = territory[Pat3(game->pat, pos)];
    }

    std::atomic_fetch_add(&statistic[pos].colors[color], 1);
    if (color == winner) {
      std::atomic_fetch_add(&statistic[pos].colors[static_cast<int>(StatisticInformation::Win)], 1);
    }
  }
  std::atomic_fetch_add(&statistic_count, 1);
}


/**
 * @~english
 * @brief Calculate criticality feature index.
 * @param[in] node UCT node.
 * @param[in] node_statistic Statistic information for UCT node.
 * @param[in] color Player's color.
 * @param[in] index Criticality feature index.
 * @~japanese
 * @brief Criticalityの特徴インデックスの計算
 * @param[in] node UCTノード
 * @param[in] node_statistic UCTノードの統計情報
 * @param[in] color 手番の色
 * @param[in] index Criticalityの特徴インデックス
 */
static void
CalculateCriticalityIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index )
{
  const int other = GetOppositeColor(color);
  const double inv_count = (statistic_count.load() > 0) ? 1.0 / statistic_count.load() : 1.0;
  const int child_num = node->child_num;
  const double win = static_cast<double>(node->win) / node->move_count;
  const double lose = 1.0 - win;
  double tmp;

  index[0] = 0;

  for (int i = 1; i < child_num; i++) {
    const int pos = node->child[i].pos;

    tmp = node_statistic[pos].colors[static_cast<int>(StatisticInformation::Win)] * inv_count
      - ((node_statistic[pos].colors[color] * inv_count) * win
         + (node_statistic[pos].colors[other] * inv_count) * lose);
    if (tmp < 0) tmp = 0;
    index[i] = static_cast<int>(tmp * CRITICALITY_TERM);
    if (index[i] > criticality_max - 1) index[i] = criticality_max - 1;
  }
}


/**
 * @~english
 * @brief Calculate criticality.
 * @param[in] color Player's color.
 * @~japanese
 * @brief Criticalityの計算
 * @param[in] color 手番の色
 */
static void
CalculateCriticality( int color )
{
  const int other = GetOppositeColor(color);
  const double inv_count = (statistic_count.load() > 0) ? 1.0 / statistic_count.load() : 1.0;
  const double win = static_cast<double>(uct_node[current_root].win) / uct_node[current_root].move_count;
  const double lose = 1.0 - win;
  double tmp;

  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];

    tmp = statistic[pos].colors[static_cast<int>(StatisticInformation::Win)] * inv_count
      - ((statistic[pos].colors[color] * inv_count) * win
         + (statistic[pos].colors[other] * inv_count) * lose);

    criticality[pos] = tmp;
    if (tmp < 0) tmp = 0;
    criticality_index[pos] = static_cast<int>(tmp * CRITICALITY_TERM);
    if (criticality_index[pos] > criticality_max - 1) criticality_index[pos] = criticality_max - 1;
  }
}


/**
 * @~english
 * @brief Calculate ownership feature index.
 * @param[in] node UCT node.
 * @param[in] node_statistic Statistic information for UCT node.
 * @param[in] color Player's color
 * @param[in] index Ownership feature index.
 * @~japanese
 * @brief Ownershipの特徴インデックスの計算
 * @param[in] node UCTノード
 * @param[in] node_statistic UCTノードの統計情報
 * @param[in] color 手番の色
 * @param[in] index Ownerの特徴インデックス
 */
static void
CalculateOwnerIndex( uct_node_t *node, statistic_t *node_statistic, int color, int *index )
{
  const double inv_count = (statistic_count.load() > 0) ? 1.0 / statistic_count.load() : 1.0;
  const int child_num = node->child_num;

  index[0] = 0;

  for (int i = 1; i < child_num; i++){
    const int pos = node->child[i].pos;
    index[i] = static_cast<int>(static_cast<double>(node_statistic[pos].colors[color]) * 10.0 * inv_count + 0.5);
    if (index[i] > OWNER_MAX - 1) index[i] = OWNER_MAX - 1;
    if (index[i] < 0)             index[i] = 0;
  }
}


/**
 * @~english
 * @brief Calculate ownership feature index.
 * @param[in] color Player's color.
 * @~japanese
 * @brief Ownershipの特徴インデックスの計算
 * @param[in] color 手番の色
 */
static void
CalculateOwner( int color )
{
  const double inv_count = (statistic_count.load() > 0) ? 1.0 / statistic_count.load() : 1.0;

  for (int i = 0; i < pure_board_max; i++){
    const int pos = onboard_pos[i];
    owner_index[pos] = static_cast<int>(static_cast<double>(statistic[pos].colors[color]) * 10.0 * inv_count + 0.5);
    if (owner_index[pos] > OWNER_MAX - 1) owner_index[pos] = OWNER_MAX - 1;
    if (owner_index[pos] < 0)             owner_index[pos] = 0;
  }
}


/**
 * @~english
 * @brief Analyze current position by UCT algorithm.
 * @param[in] game Current board position data.
 * @param[in] color Player's color.
 * @return Current black player's score.
 * @~japanese
 * @brief UCTアルゴリズムによる局面解析
 * @param[in] game 現在の局面情報
 * @param[in] color 手番の色
 * @return 黒番の地
 */
int
UctAnalyze( game_info_t *game, int color )
{
  std::thread *worker[THREAD_MAX];

  // 探索情報をクリア
  for (int i = 0; i < board_max; i++) {
    statistic[i].clear();
  }
  statistic_count = 0;
  std::fill_n(criticality_index, board_max, 0);  
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;
  }
  ResetPoCount();

  ClearUctHash();

  current_root = ExpandRoot(game, color);

  SetPoHalt(10000);

  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    t_arg[i].lz_analysis_cs = -1;
    worker[i] = new std::thread(ParallelUctSearch, &t_arg[i]);
  }

  for (int i = 0; i < threads; i++) {
    worker[i]->join();
    delete worker[i];
  }

  int black = 0, white = 0;

  const int inv_count = 1.0 / statistic_count.load();
  for (int y = board_start; y <= board_end; y++) {
    for (int x = board_start; x <= board_end; x++) {
      const int pos = POS(x, y);
      const double ownership_value = static_cast<double>(statistic[pos].colors[S_BLACK]) * inv_count;
      if (ownership_value > 0.5) {
        black++;
      } else {
        white++;
      }
    }
  }

  PrintOwner(&uct_node[current_root], statistic, color, statistic_count.load(), owner);

  return black - white;
}


/**
 * @~english
 * @brief Copy ownership values.
 * @param[out] dest Copy destination.
 * @~japanese
 * @brief Ownerの値のコピー
 * @param[out] dest コピー先
 */
void
OwnerCopy( int *dest )
{
  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    dest[pos] = static_cast<int>(static_cast<double>(statistic[pos].colors[my_color]) / uct_node[current_root].move_count * 100);
  }
}


/**
 * @~english
 * @brief Copy criticality value.
 * @param[out] dest Copy destination.
 * @~japanese
 * @brief Criticalityの値のコピー
 * @param[out] dest コピー先
 */
void
CopyCriticality( double *dest )
{
  for (int i = 0; i < pure_board_max; i++) {
    const int pos = onboard_pos[i];
    dest[pos] = criticality[pos];
  }
}


/**
 * @~english
 * @brief Copy statistic information of Monte-Carlo simulation.
 * @param[out] dest Copy destination.
 * @~japanese
 * @brief 統計情報のコピー
 * @param[out] dest コピー先
 */
void
CopyStatistic( statistic_t *dest )
{
  for (int i = 0; i < board_max; i++) {
    dest[i] = statistic[i];
  }
}


/**
 * @~english
 * @brief Generate next move by UCT search for KGS clean up mode.
 * @param[in] game Current board position data.
 * @param[in] color Player's color.
 * @return Coordinate of next move.
 * @~japanese
 * @brief UCT探索による着手生成 (KGSの終局処理用)
 * @param[in] game 現在の局面情報
 * @param[in] color 手番の色
 * @return 次の着手の座標
 */
int
UctSearchGenmoveCleanUp( game_info_t *game, int color )
{
  int pos;
  double wp;
  std::thread *worker[THREAD_MAX];

  for (int i = 0; i < board_max; i++) {
    statistic[i].clear();
  }
  statistic_count = 0;
  std::fill_n(criticality_index, board_max, 0); 
  for (int i = 0; i < board_max; i++) {
    criticality[i] = 0.0;
  }

  StartTimer();

  ResetPoCount();

  current_root = ExpandRoot(game, color);

  if (uct_node[current_root].child_num <= 1) {
    return PASS;
  }

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];
    owner[pos] = 50.0;
  }

  SetPoHalt(GetPoNum());

  DynamicKomi(game, &uct_node[current_root], color);

  for (int i = 0; i < threads; i++) {
    t_arg[i].thread_id = i;
    t_arg[i].game = game;
    t_arg[i].color = color;
    t_arg[i].lz_analysis_cs = -1;
    worker[i] = new std::thread(ParallelUctSearch, &t_arg[i]);
  }

  for (int i = 0; i < threads; i++) {
    worker[i]->join();
    delete worker[i];
  }

  child_node_t *uct_child = uct_node[current_root].child;

  const int select_index = SelectMaxVisitChild(uct_node[current_root]);

  const double finish_time = CalculateElapsedTime();

  wp = (double)uct_node[current_root].win / uct_node[current_root].move_count;

  const int po_speed = static_cast<int>(CalculatePlayoutSpeed(finish_time, threads));

  PrintPlayoutInformation(&uct_node[current_root], po_speed, finish_time, 0);
  PrintOwner(&uct_node[current_root], statistic, color, statistic_count.load(), owner);

  PrintBestSequence(game, uct_node, current_root, color);

  CalculateNextPlayouts(game, color, wp, finish_time, threads);

  int count = 0;

  for (int i = 0; i < pure_board_max; i++) {
    pos = onboard_pos[i];

    if (owner[pos] >= 5 && owner[pos] <= 95) {
      candidates[pos] = true;
      count++;
    } else {
      candidates[pos] = false;
    }
  }

  if (count == 0) {
    pos = PASS;
  } else {
    pos = uct_child[select_index].pos;
  }
  
  if (static_cast<double>(uct_child[select_index].win) / uct_child[select_index].move_count < RESIGN_THRESHOLD) {
    return PASS;
  } else {
    return pos;
  }
}


/**
 * @~english
 * @brief Correct descendent node indices.
 * @param[in, out] indexes Descendent node indices.
 * @param[in] index Node index.
 * @~japanese
 * @brief 子孫ノードのインデックスの収集
 * @param[in, out] indexes 子孫ノードのインデックス
 * @param[in] index 現在のインデックス
 */
static void
CorrectDescendentNodes( std::vector<int> &indexes, int index )
{
  child_node_t *uct_child = uct_node[index].child;
  const int child_num = uct_node[index].child_num;

  indexes.push_back(index);

  for (int i = 0; i < child_num; i++) {
    if (uct_child[i].index != NOT_EXPANDED) {
      CorrectDescendentNodes(indexes, uct_child[i].index);
    }
  }   
}


/**
 * @~english
 * @brief Get visits threshold for node expansion.
 * @param[in] game Board position data.
 * @return Visit threshold for node expansion.
 * @~japanese
 * @brief ノードを展開する探索回数の閾値の取得
 * @param[in] game 局面情報
 * @return ノードを展開する探索回数の閾値
 */
static int
GetExpandThreshold( const game_info_t *game )
{
  if (game->moves > 2 &&
      game->record[game->moves - 1].pos == PASS &&
      game->record[game->moves - 2].pos == PASS) {
    return INT_MAX - 1;
  } else {
    return expand_threshold;
  }
}

/**
 * @brief NNの推論結果（Policy）を用いてノードを実際に展開する
 */
int ExpandNodeWithNN(game_info_t *game, int color, int current, int child_index, at::Tensor policy) {
    std::lock_guard<std::mutex> lock(mutex_expand);

    // すでに展開済みなら、何もしない（gameはここでは消さない）
    if (uct_node[current].child[child_index].index != NOT_EXPANDED) {
        return uct_node[current].child[child_index].index;
    }

    int new_index = ExpandNode(game, color, current, policy); 
    uct_node[current].child[child_index].index = new_index;
    
    // ここで FreeGame(game) はしない！
    return new_index;
}
/**
 * @brief NNの推論結果（Value）をルート方向へ報告し、Virtual Lossを解除する
 */
// 修正案：探索経路（path）を受け取るように変更する必要があります
void BackpropagateNNResult(std::vector<std::pair<int, int>> &path, double win_rate) {
    // パスを逆順に辿って各ノードの勝率を更新する
    // 深さ1の場合は path[0] だけが対象になる
    // BackpropagateNNResult 内
    fprintf(stderr, "NN Result Backpropagated: Value=%.4f, PathSize=%zu\n", win_rate, path.size());
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        int node_idx = it->first;
        int child_idx = it->second;

        // Virtual Lossの解除
        uct_node[node_idx].child[child_idx].virtual_loss.fetch_sub(1);

        // 真のValueで統計を更新
        UpdateNodeStats(node_idx, child_idx, win_rate);
        
        // 手番が入れ替わるので win_rate を反転
        win_rate = 1.0 - win_rate;
    }
}