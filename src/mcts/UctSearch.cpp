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
// #include "mcts/Simulation.hpp"
// #include "mcts/UctRating.hpp"
#include "mcts/UctSearch.hpp"
#include "mcts/ucb/UCBEvaluation.hpp"
#include "util/Utility.hpp"
#include "mcts/BatchQueue.hpp" // 先ほど作成したヘッダーをインクルード

static volatile bool interrupted = false;

#if defined (_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#endif

#ifndef NN_BATCH_SIZE
#define NN_BATCH_SIZE 16
#endif

#ifndef MCTS_TREE_SIZE
#define MCTS_TREE_SIZE 65536 // Python版のデフォルト値
#endif

#ifndef MAX_NODES
#define MAX_NODES (PURE_BOARD_SIZE * PURE_BOARD_SIZE + 1)
#endif

#if !defined(CRITICALITY_MAX)
  #define CRITICALITY_MAX 100
  #define CRITICALITY_INTERVAL 100
  #define CRITICALITY_TERM 1.0
  #define OWNER_MAX 100
  #define UCT_INDEX_MAX 100
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

BatchQueue g_batch_queue; 

extern double const_playout;

extern int threads;

std::mutex model_mutex;

std::atomic<int> uct_node_count(1);      // 実体定義

extern torch::Device device;             // 外部参照宣言

extern torch::jit::script::Module tamago_model;

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

// ルートの展開
static int ExpandRoot( game_info_t *game, int color );

// UCT探索
static void ParallelUctSearch( thread_arg_t *arg );

// UCT探索(予測読み)
static void ParallelUctSearchPondering( thread_arg_t *arg );

static int RateComp( const void *a, const void *b );

// UCB値が最大の子ノードを返す
static int SelectMaxUcbChild( uct_node_t &node, const int moves, const int color, std::mt19937_64 &mt );

// 各座標の統計処理
static void Statistic( game_info_t *game, int winner );

// UCT探索(1回の呼び出しにつき, 1回の探索)
static double UctSearch(game_info_t *game, int color, std::mt19937_64 &mt, int current, std::vector<std::pair<int, int>> &path);

// ノード展開の閾値を取得
static int GetExpandThreshold( const game_info_t *game );

// 2. BatchHandler.cpp にある関数の宣言
extern void ProcessMiniBatch(torch::jit::script::Module &model, torch::Device &device);

extern void IncrementPoCount(void);          

int UctSearchGenmove(struct game_info_t *game, int color, int mode);

void GetPrincipalVariation(int current, std::vector<int> &pv);

int ExpandNodeWithNN(game_info_t *game, int color, int current, int child_index, torch::Tensor policy_tensor);

extern void UpdateNodeStats(int node_idx, int child_idx, double v);

extern int GetPlayoutLimit(void);             // po_limit取得用

extern void IncrementPoCount(void);           // Atomicの代わり

extern void PrintSearchStatus( int root_index );


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
 * @brief 探索の終了条件を管理する
 */
struct SearchMonitor {
    std::chrono::system_clock::time_point start_time;
    double time_limit;       // 秒単位 (例: 0.6)
    int visit_threshold;    // 最大探索回数 (例: 1600)

    /**
     * @brief 打ち切り判定（メインループから呼ぶ）
     */
    bool IsSearchTerminated(int root_index) {
        uct_node_t &root = uct_node[root_index];
        int total_visits = root.move_count.load();

        // 1. 時間超過
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;
        if (elapsed.count() > time_limit) return true;

        // 2. 規定の探索回数に到達
        if (total_visits >= visit_threshold) return true;

        // 3. 逆転不可能判定 (1位と2位の訪問数差を確認)
        if (root.child_num < 2) return false;
        int first_v = 0, second_v = 0;
        for (int i = 0; i < root.child_num; ++i) {
            int v = root.child[i].move_count.load();
            if (v > first_v) { second_v = first_v; first_v = v; }
            else if (v > second_v) { second_v = v; }
        }
        int remaining = visit_threshold - total_visits;
        if (first_v > second_v + remaining) return true;

        return false;
    }
};

/**
 * @brief 現在の探索木から最善応手系列(PV)を取得する
 */
void GetPrincipalVariation(int current, std::vector<int> &pv) {
    // 10手以上、または異常なノードインデックスの場合は終了
    if (pv.size() >= 10 || current < 0) return;

    auto& node = uct_node[current];
    int max_visits = -1;
    int best_child_idx = -1;

    // 最も多く探索された手（信頼できる手）を探す
    for (int i = 0; i < node.child_num; i++) {
        int v = node.child[i].move_count; 
        if (v > max_visits) {
            max_visits = v;
            best_child_idx = i;
        }
    }

    if (best_child_idx != -1 && max_visits > 0) {
        int pos = node.child[best_child_idx].pos;
        int next_node = node.child[best_child_idx].index;
        
        pv.push_back(pos);

        // 次のノードが展開されており、インデックスが妥当なら再帰
        if (next_node != NOT_EXPANDED && next_node >= 0) {
            GetPrincipalVariation(next_node, pv);
        }
    }
}

/**
 * @brief 思考エンジン本体（メインループ）
 * @param[in] game 現在の局面
 * @param[in] color コンピュータの手番の色
 * @return 選択された着手（pos）
 */
int
UctSearchGenmove(game_info_t *game, int color, int mode)
{
    std::cerr << "LOG: UctSearchGenmove started." << std::endl;

    torch::NoGradGuard no_grad;

    // モデルのデバイス転送（初回のみ）
    static bool is_model_prepared = false;
    if (!is_model_prepared) {
        tamago_model.to(device);
        tamago_model.eval();
        is_model_prepared = true;
    }

    // 探索前の初期化
    ClearUctHash();
    ResetPoCount();

    int root = 0; // 初期値

    // --- 3. ルート展開 ---
    if (uct_node[root].child_num == 0) {
        at::Tensor input = TamaGoFeature::GenerateInputPlanes(game, color);
        
        // policyをスコープ外で使うために宣言
        at::Tensor policy;
        
        {            
            // forwardの結果を一度IValueで受け取り、Tupleのポインタへ変換
            auto output_ivalue = tamago_model.forward({input.to(device)});
            auto elements = output_ivalue.toTuple();
            
            // policyを取り出す (ポインタアクセスなので -> を使用)
            policy = elements->elements()[0].toTensor().softmax(1).to(torch::kCPU).view({-1});
        } // ここでアンロック

        // アンロックされた状態でノードを展開
        root = ExpandNodeWithNN(game, color, -1, 0, policy); 
    }

    fprintf(stderr, "DEBUG: Root expanded. index=%d, child_num=%d\n", root, uct_node[root].child_num);

    if (root == -1 || uct_node[root].child_num == 0) {
        fprintf(stderr, "ERROR: Failed to expand root node!\n");
        return PASS;
    }

    // --- 4. 探索スレッド起動 ---
    int thread_num = 1;
    std::vector<std::thread*> workers(thread_num);
    std::atomic<bool> interrupted(false);

    for (int i = 0; i < thread_num; i++) {
        workers[i] = new std::thread([&, color, root]() {
            std::mt19937_64 local_mt(std::random_device{}());
            game_info_t *copy_game = AllocateGame();
            
            while (!interrupted) {
                // 停止条件（プレイアウト数など）
                if (GetPoCount() >= 1000) break;

                CopyGame(copy_game, game);
                std::vector<std::pair<int, int>> path;
                
                // 1. 探索実行（ここでノードの win/move_count が更新される）
                float result = UctSearch(copy_game, color, local_mt, root, path);
                
                // 2. プレイアウト数を加算（これがないとメインループが止まらない）
                IncrementPoCount();
            }
            FreeGame(copy_game);
        });
    }

    // スレッドの終了を待機
    for (int i = 0; i < thread_num; i++) {
        if (workers[i] != nullptr && workers[i]->joinable()) {
            workers[i]->join();
            delete workers[i];
        }
    }

    // --- 5. ログ出力と着手決定 (変更なし) ---
    // --- 5. ログ出力の修正 ---
    fprintf(stderr, "--- Root Node Stats (My View: %s) ---\n", color == S_BLACK ? "Black" : "White");
    for (int i = 0; i < uct_node[root].child_num; ++i) {
        auto& c = uct_node[root].child[i];
        if (c.move_count > 0) {
            // 子ノードは相手番の評価値(1.0-v)を持っているので、
            // もう一度 1.0 から引くことで自分の視点に戻す
            double my_wp = (double)c.win / c.move_count; 
            fprintf(stderr, "pos=%3d n=%4d win_raw=%7.2f my_wp=%.4f\n", 
                    c.pos, (int)c.move_count, (double)c.win, my_wp);
        }
    }

    double best_wp = 0.0;
    int select_pos = SelectMove(game, uct_node[root], color, best_wp);

    std::vector<int> pv;
    GetPrincipalVariation(root, pv);

    fprintf(stderr, "--- Move Selection Decision ---\n");
    fprintf(stderr, " Best Move WP: %.4f\n", best_wp);
    
    double pass_wp = 0.0;
    if (uct_node[root].child_num > 0 && uct_node[root].child[0].pos == PASS) {
        pass_wp = (double)uct_node[root].child[0].win / std::max(1, (int)uct_node[root].child[0].move_count);
    }
    fprintf(stderr, " Pass WP      : %.4f\n", pass_wp);

    fprintf(stderr, " PV: ");
    for (int p : pv) {
        char pos_str[10];
        IntegerToString(p, pos_str);
        fprintf(stderr, "%s ", pos_str);
    }
    fprintf(stderr, "\n-------------------------------\n");

    return select_pos;
}


void UpdateNodeStats(int current, int child_index, double v) {
    if (current == 0) { // ルート直下の枝だけ監視
        std::cerr << "Update Root Child[" << child_index << "] v=" << v 
                  << " total_win=" << uct_node[current].child[child_index].win 
                  << " n=" << uct_node[current].child[child_index].move_count << std::endl;
    }
    // 枝の訪問数 +1
    uct_node[current].child[child_index].move_count.fetch_add(1);
    
    // 勝率の加算
    auto& atomic_child_win = reinterpret_cast<std::atomic<double>&>(uct_node[current].child[child_index].win);
    double old_child_win = atomic_child_win.load();
    while (!atomic_child_win.compare_exchange_weak(old_child_win, old_child_win + v));

    // 親ノード全体の訪問数 +1 (ルートのVisits管理に必須)
    uct_node[current].move_count.fetch_add(1);
}

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
 * NN Policyを用いてノードを展開し、そのインデックスを返す
 * game: 現在の盤面状態
 * color: 手番の色
 * policy_tensor: NNが出力したPolicyテンソル
 */
int
ExpandNodeWithNN(game_info_t *game, int color, int current, int child_index, torch::Tensor policy_tensor)
{
    const int moves = game->moves;
    const unsigned long long hash = game->move_hash;

    // 1. まずハッシュテーブルを確認（ロックなし）
    unsigned int index = FindSameHashIndex(hash, color, moves);
    
    // まだテーブルにない場合のみ、展開処理に入る
    if (index == uct_hash_size) {
        
        // --- 重い処理はロックの外で行う ---
        // Tensorの転送をロックの外に出す
        auto policy_cpu = policy_tensor.to(torch::kCPU).view({-1});
        float* p_ptr = policy_cpu.data_ptr<float>();

        // 候補手の計算（スタック上に一時保存）
        struct TempCandidate { int pos; float policy; };
        std::vector<TempCandidate> candidates;
        candidates.reserve(82);

        for (int i = 0; i < 81; i++) {
            const int ray_pos = onboard_pos[i];
            if (game->board[ray_pos] == S_EMPTY && IsLegal(game, ray_pos, color)) {
                candidates.push_back({ray_pos, p_ptr[i]});
            }
        }
        if (IsLegal(game, PASS, color)) {
            candidates.push_back({PASS, p_ptr[81]});
        }

        {

            // 二重チェック（ロックを待っている間に他スレッドが作成した可能性があるため）
            index = FindSameHashIndex(hash, color, moves);
            if (index == uct_hash_size) {
                index = SearchEmptyIndex(hash, color, moves);
                if (index == uct_hash_size) return -1;

                int pm1 = (moves > 0) ? game->record[moves - 1].pos : PASS;
                int pm2 = (moves > 1) ? game->record[moves - 2].pos : PASS;
                InitializeNode(uct_node[index], pm1, pm2);

                child_node_t *uct_child = uct_node[index].child;
                int child_num = 0;

                // 準備しておいた候補手をノードにコピー
                for (const auto& cand : candidates) {
                    InitializeCandidate(uct_child[child_num], child_num, cand.pos, false);
                    uct_child[child_num].nn_policy = cand.policy; 
                    child_num++;
                }

                uct_node[index].child_num = child_num;
                // 初期化完了フラグとして機能
                uct_node[index].width = 1; 
            }
        } // ここでロック解除
    }

    // 2. 親ノードとの接続が必要な場合（探索中）
    if (current != -1 && child_index != -1) {
        // 親ノードの特定の枝に子インデックスを書き込む
        // ここはアトミックな書き込み、または別のロックが必要な場合があります
        uct_node[current].child[child_index].index = (int)index;
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
void
RatingNode( game_info_t *game, int color, int current )
{
  // --- 古いロジックはすべて削除 ---
  
  uct_node_t &node = uct_node[current];
  child_node_t *child = node.child;

  // すでに ExpandNodeWithNN 等で policy がセットされているはずなので
  // ここでは古い 'rate' (バイアス) を 0 にリセットするだけでOK
  for (int i = 0; i < node.child_num; i++) {
    child[i].rate = 0.0; 
  }

  // もし特定の「手」に対してNN以外で強制的にバイアスをかけたい場合のみ、ここに書く
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
  std::cerr << "LOG: Thread " << arg->thread_id << " started." << std::endl;

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


void SetNodeStatistics(int node_index, const torch::Tensor& policy_tensor, double value) {
    if (node_index < 10) {
        std::cerr << "NODE=" << node_index << " RAW_VALUE=" << value << std::endl;
    }
    // 1. ノード自体の初期評価値をセット
    // ※現在のuct_node_tの構造に合わせて、win(勝利数)などを初期化
    uct_node[node_index].win = value; // 初期値は0.0、後でプレイアウト結果で加算される
    uct_node[node_index].move_count = 0; // 1回探索したことにするか、0にするかは実装方針次第

    // 2. Policyテンソルから各枝へ確率を配分
    // CPUへ転送し、アクセス可能なポインタを取得
    auto policy_cpu = policy_tensor.to(torch::kCPU);
    float* p_policy = policy_cpu.data_ptr<float>();
    
    int child_num = uct_node[node_index].child_num;
    for (int i = 0; i < child_num; ++i) {
        int pos = uct_node[node_index].child[i].pos;
        
        // 囲碁の座標体系（19x19なら361がパス）と一致しているか確認が必要
        // 一般的には NN出力の 361番目がパス
        float p_val;
        if (pos == PASS) {
            p_val = p_policy[pure_board_size * pure_board_size];
        } else {
            // 2次元座標を1次元インデックスに変換（NNの入力形式に合わせる）
            int x = X(pos);
            int y = Y(pos);
            p_val = p_policy[(y - OB_SIZE) * pure_board_size + (x - OB_SIZE)];
        }
        
        // UctSearch.cpp で定義されている変数名 (nn_policy) に合わせる
        uct_node[node_index].child[i].nn_policy = p_val;
    }
}


/**
 * @brief 未展開の空ノードを確保し、インデックスを返す
 * @return 確保したノードのインデックス
 */
static int CreateEmptyNode(void) {
    int new_index = uct_node_count.fetch_add(1);

    if (new_index >= (int)UCT_HASH_SIZE) { // 型比較の警告対策
        std::cerr << "MCTS Tree Overflow!" << std::endl;
        exit(1);
    }

    uct_node[new_index].move_count.store(0);
    uct_node[new_index].win.store(0.0);
    uct_node[new_index].child_num = 0;

    for (int i = 0; i < MAX_NODES; ++i) {
        // index が std::atomic<int> でない場合、直接代入
        uct_node[new_index].child[i].index = NOT_EXPANDED;
        uct_node[new_index].child[i].move_count.store(0);
        uct_node[new_index].child[i].virtual_loss.store(0);
    }
    return new_index;
}

/**
 * @brief UCBを利用したモンテカルロ木探索
 * @return 0.0-1.0: 終局による確定値, -1.0: バッチ処理待ち（更新スキップ）
 */
/**
 * @brief UCBを利用したモンテカルロ木探索 (Python非同期版の移植)
 */
/**
 * @brief UCBを利用したモンテカルロ木探索 (再帰・一元更新版)
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
        // 戻り値は 0.0 ~ 1.0 (自手番視点)
        return (color == S_BLACK) ? (score > 0 ? 1.0 : 0.0) : (score < 0 ? 1.0 : 0.0);
    }

    // 2. 次の一手を選択
    int next_index = SelectMaxUcbChild(uct_node[current], game->moves, color, mt);
    int pos = uct_node[current].child[next_index].pos;

    // 3. Virtual Loss の加算
    uct_node[current].child[next_index].virtual_loss.fetch_add(1);
    path.push_back({current, next_index});

    // 4. 次ノードの準備
    int threshold = GetExpandThreshold(game);
    int current_visits = uct_node[current].child[next_index].move_count.load() +
                         uct_node[current].child[next_index].virtual_loss.load();

    double v = -1.0;

    // 1. まず現在の枝に登録されているインデックスを取得
    int next_node_idx = uct_node[current].child[next_index].index;
    int next_color = 3 - color;

    // 局面を進める準備
    auto next_game = std::make_unique<game_info_t>(); 
    CopyGame(next_game.get(), game);
    PutStone(next_game.get(), pos, color);

    unsigned long long next_hash = next_game->move_hash;

    if (current_visits < threshold + 1) {
        // --- 展開フェーズ ---
        
        // すでにハッシュテーブルにあるか確認（別ルートからの合流をチェック）
        if (next_node_idx == NOT_EXPANDED) {
            next_node_idx = (int)FindSameHashIndex(next_hash, next_color, next_game->moves);
        }

        // テーブルにもない場合は、新しく展開を依頼
        if (next_node_idx == (int)uct_hash_size) {
            // 特徴量生成
            auto input_planes = TamaGoFeature::GenerateInputPlanes(next_game.get(), next_color);
            
            // キューに積む (引数は4つ: tensor, game_ptr, path, color)
            g_batch_queue.push(input_planes, next_game.get(), path, next_color);

            if (g_batch_queue.size() >= 1) {
                ProcessMiniBatch(tamago_model, device);
            }

            // ★重要: NNスレッドがハッシュテーブルに登録するまで待機
            int wait_count = 0;
            while (true) {
                next_node_idx = (int)FindSameHashIndex(next_hash, next_color, next_game->moves);
                if (next_node_idx != (int)uct_hash_size) break; // 見つかったら抜ける

                std::this_thread::yield();
                wait_count++;
                if (wait_count % 1000000 == 0 && g_batch_queue.size() > 0) {
                    ProcessMiniBatch(tamago_model, device);
                }
            }
        }
        
        // 親の枝に確定したインデックスを接続
        uct_node[current].child[next_index].index = next_node_idx;

        // NNが書き込んだ勝率を取得
        double v_black = uct_node[next_node_idx].win.load();
        std::cerr << "DEBUG NN Value: " << v_black << std::endl;
        v = (color == S_BLACK) ? v_black : (1.0 - v_black);

    } else {
        // --- 深化フェーズ ---
        v = UctSearch(next_game.get(), next_color, mt, next_node_idx, path);
    }

    // --- 5. バックプロパゲーション ---
    
    // 仮想損失の解除
    uct_node[current].child[next_index].virtual_loss.fetch_sub(1);

    // 統計更新
    UpdateNodeStats(current, next_index, 1.0 - v);

    return 1.0 - v;
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
    
    const int total_visits = node.move_count.load();
    // 仮想損失も含めたトータルの分母で計算する
    const double sqrt_total = std::sqrt(static_cast<double>(total_visits) + 1.0);
    const double C_PUCT = 8.0; 

    double parent_q = 0.5;
    if (total_visits > 0) {
        // 既に手番側の勝率として保存されているなら、色による反転は不要！
        parent_q = static_cast<double>(node.win) / total_visits;
    }
    
    const double fpu_reduction = 0.1; 
    double fpu_value = parent_q - fpu_reduction;
    if (fpu_value < 0.0) fpu_value = 0.0;

    for (int i = 0; i < node.child_num; i++) {
        child_node_t &child = node.child[i];
        const int vis = child.move_count.load();
        const int vls = child.virtual_loss.load();
        const int n = vis + vls;
        
        // 修正案
        double q_value = fpu_value;
        if (n > 0) {
            // child.win が常に「黒の勝ち数」を指していると仮定する場合のみ反転が必要
            // もし UctSearch の再帰で 1.0 - v しているなら、ここは単純に wins / n で良い
            q_value = static_cast<double>(child.win) / n; 
        }

        // 2. U値 (Policy項)
        double u_value = C_PUCT * child.nn_policy * sqrt_total / (1.0 + n);
        
        double score = q_value + u_value;

        if (score > max_score) {
            max_score = score;
            select_index = i;
        } else if (score == max_score) {
            if (mt() % 2 == 0) select_index = i;
        }
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
UctSearchGenmoveCleanUp(game_info_t *game, int color)
{
    int pos;
    double wp;
    std::thread *worker[THREAD_MAX]; // ローカルで配列を確保（またはグローバルを使用）

    // --- 1. 初期化 ---
    for (int i = 0; i < board_max; i++) statistic[i].clear();
    statistic_count = 0;
    std::fill_n(criticality_index, board_max, 0); 
    for (int i = 0; i < board_max; i++) criticality[i] = 0.0;

    StartTimer();
    ResetPoCount();

    // current_rootを更新（rootという変数名で統一すると分かりやすいです）
    int root = ExpandRoot(game, color); 
    if (uct_node[root].child_num <= 1) return PASS;

    for (int i = 0; i < pure_board_max; i++) owner[onboard_pos[i]] = 50.0;

    // 探索制限の設定
    int po_limit = GetPoNum();
    SetPoHalt(po_limit);
    DynamicKomi(game, &uct_node[root], color);

    // 終了判定用のモニタ
    SearchMonitor monitor;
    monitor.start_time = std::chrono::system_clock::now();
    monitor.visit_threshold = po_limit;
    monitor.time_limit = 6.0; // 秒

    // --- 4. 探索開始 ---
    interrupted = false;
    int thread_count = static_cast<int>(threads);
    for (int i = 0; i < thread_count; i++) {
        worker[i] = new std::thread([&, i, root]() {
            std::mt19937_64 local_mt(std::random_device{}());
            while (!interrupted) {
                // メインスレッドが止めるまで、あるいはPO数に達するまで探索
                if (monitor.IsSearchTerminated(root)) break;

                game_info_t *copy_game = AllocateGame();
                CopyGame(copy_game, game);
                std::vector<std::pair<int, int>> path;
                UctSearch(copy_game, color, local_mt, root, path);
                FreeGame(copy_game);
                IncrementPoCount();
            }
        });
    }

    // ★追加: メインスレッドでキューを消化するループ
    while (GetPoCount() < 10) { // まずは10回程度でテスト
        if (!g_batch_queue.empty()) {
            ProcessMiniBatch(tamago_model, device);
        } else {
            std::this_thread::yield(); // キューが空なら少し待つ
        }
    }
    interrupted = true; // スレッドに停止を指示

    // スレッドをすべて回収
    for (int i = 0; i < thread_count; i++) {
        if (worker[i] && worker[i]->joinable()) {
            worker[i]->join();
            delete worker[i];
            worker[i] = nullptr;
        }
    }

    // ★ 残ったキューをすべて処理しきる
    int empty_wait = 0;
    while (empty_wait < 20) {
        if (g_batch_queue.size() > 0) {
            ProcessMiniBatch(tamago_model, device);
            empty_wait = 0;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            empty_wait++;
        }
    }

    // --- 7. 結果の集計 ---
    child_node_t *uct_child = uct_node[root].child;
    const int select_index = SelectMaxVisitChild(uct_node[root]);
    const double finish_time = CalculateElapsedTime();

    wp = (double)uct_node[root].win / std::max(1, (int)uct_node[root].move_count);

    // ログ出力など
    const int po_speed = static_cast<int>(CalculatePlayoutSpeed(finish_time, threads));
    PrintPlayoutInformation(&uct_node[root], po_speed, finish_time, 0);
    PrintBestSequence(game, uct_node, root, color);

    pos = uct_child[select_index].pos;
    return pos;
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



