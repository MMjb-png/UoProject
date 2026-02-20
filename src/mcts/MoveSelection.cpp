/**
 * @file MoveSelection.cpp
 * @author Yuki Kobayashi
 * @~english
 * @brief Move selection from search result.
 * @~japanese
 * @brief 着手選択処理
 */
#include <iostream>
#include <iomanip>
#include "common/Message.hpp"
#include "feature/Territory.hpp"
#include "mcts/MoveSelection.hpp"
#include "mcts/UctSearch.hpp"


/**
 * @~english
 * @brief Winning rate threshold for resignation.
 * @~japanese
 * @brief 投了する勝率の閾値
 */
static double resign_threshold = RESIGN_THRESHOLD;

/**
 * @~english
 * @brief Capturing all opponent's dead stones mode.
 * @~japanese
 * @brief 相手の全ての死に石を打ち上げるモード
 */
static bool capture_all_mode = false;


/**
 * @~english
 * @brief Set capturing all opponent's dead stones mode.
 * @param[in] flag Activation flag.
 * @~japanese
 * @brief 相手の全ての死に石を打ち上げるモードの設定
 * @param[in] flag 有効化フラグ
 */
void
SetCaptureAllMode( const bool flag )
{
  capture_all_mode = flag;
}


/**
 * @~english
 * @brief Set Winning rate threshold for resignation.
 * @param[in] threshold Winning rate threshold for resignation.
 * @~japanese
 * @brief 投了する勝率の閾値の設定
 * @param[in] threshold 投了する勝率の閾値
 */
void
SetResignThreshold( const double threshold )
{
  if (threshold <= 0.0) {
    PrintResignThresholdIsTooSmall(threshold);
    resign_threshold = 0.0;
  } else if (threshold > 1.0) {
    PrintResignThresholdIsTooLarge(threshold);
  } else {
    resign_threshold = threshold;
  }
}


/**
 * @~english
 * @brief Select move from max visited move.
 * @param[in] root Root node.
 * @return Next move.
 * @~japanese
 * @brief 探索回数最大の手を選択
 * @param[in] root ルート
 * @return 次の着手
 */
int
SelectMaxVisitChild( const uct_node_t &root )
{
  const child_node_t *child = root.child;
  int select_index = PASS_INDEX;
  int max_count = child[PASS_INDEX].move_count;

  for (int i = 1; i < root.child_num; i++) {
    if (child[i].move_count > max_count) {
      select_index = i;
      max_count = child[i].move_count;
    }
  }

  return select_index;
}


int
SelectMove( const game_info_t *game, const uct_node_t &root, const int color, double &best_wp )
{
  const child_node_t *child = root.child;
  const int select_index = SelectMaxVisitChild(root);
  // パスの評価値を計算
  const double pass_wp = CalculatePassWinningPercentage(root);
  
  // 最善手の勝率を計算
  best_wp = (child[select_index].move_count > 0) 
            ? static_cast<double>(child[select_index].win) / child[select_index].move_count 
            : 0.0;

  // --- デバッグ比較出力 ---
  std::cerr << "--- Move Selection Decision ---" << std::endl;
  std::cerr << " Best Move WP: " << std::fixed << std::setprecision(4) << best_wp << std::endl;
  std::cerr << " Pass WP     : " << pass_wp << std::endl;
  std::cerr << "-------------------------------" << std::endl;

  // 1. 投了判定
  if (best_wp < resign_threshold && game->moves > 20) {
    return RESIGN;
  }

  // 2. 手数上限
  if (game->moves >= MAX_MOVES) return PASS;

  // 3. パス判定の修正
  // 序盤（moves < 10）は、pass_wpがどれだけ高くても強制的に盤上に打たせる
  if (game->moves < 10) {
      if (child[select_index].pos == PASS && root.child_num > 1) {
          // パスが最大訪問数でも、2番目に訪問数が多い手（盤上の手）を探す
          int second_index = (select_index == 0) ? 1 : 0;
          return child[second_index].pos;
      }
      return child[select_index].pos;
  }

  // 4. Ray本来の終局ロジック
  if (pass_wp >= PASS_THRESHOLD && (game->record[game->moves - 1].pos == PASS)) {
    return PASS;
  } else if (game->moves > 3 &&
             game->record[game->moves - 1].pos == PASS &&
             game->record[game->moves - 3].pos == PASS) {
    return PASS;
  }

  return child[select_index].pos;
}
