import random
import numpy as np
from typing import Tuple, List,Any
import pandas as pd

specium_cost = 5
class Player:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.charge: int = 0
        self.last_action: str = None

    def charge_point(self) -> None:
        """チャージを1溜める"""
        self.charge += 1
        self.last_action = "チャージ"

    def beam_attack(self, target: "Player") -> None:
        """ビーム攻撃: チャージ1消費"""
        if self.charge >= 1:
            self.charge -= 1
            self.last_action = "ビーム"

    def specium_attack(self, target: "Player") -> None:
        """スペシウム光線攻撃: チャージ{specium_cost}消費"""
        if self.charge >= specium_cost:
            self.charge -= specium_cost
            self.last_action = "スペシウム光線"

    def guard(self) -> None:
        """ガード: 防御"""
        self.last_action = "ガード"

class CCGameEnv:
    """CCレモンゲームの環境"""
    def __init__(self) -> None:
        # プレイヤー1とプレイヤー2の初期設定
        self.player1 = Player("Player 1")
        self.player2 = Player("Player 2")

    def reset(self) -> Tuple[int, int, int, int]:
        """ゲームを初期状態にリセット"""
        self.player1.charge = 0
        self.player2.charge = 0
        self.player1.last_action = None
        self.player2.last_action = None
        return self.get_state()

    def get_state(self) -> Tuple[int, int, int, int]:
        """現在のゲームの状態を返す"""
        return (self.player1.charge, self.player2.charge)

    def step(self, action1: int, action2: int) -> Tuple[Tuple[int, int, int, int], int, bool]:
        """プレイヤー1とプレイヤー2の行動を実行"""
        # プレイヤー1の行動
        self._take_action(self.player1, action1, self.player2)
        # プレイヤー2の行動
        self._take_action(self.player2, action2, self.player1)
        # 勝敗判定
        result = self._check_winner(self.player1, self.player2)
        if result == 1:  # プレイヤー1の勝ち
            return self.get_state(), 1, True
        elif result == -1:  # プレイヤー2の勝ち
            return self.get_state(), -1, True
        return self.get_state(), 0, False # ゲーム続行

    def _take_action(self, player: "Player", action: int, opponent: "Player") -> None:
        """プレイヤーのアクションを実行"""
        if action == 0:  # チャージ
            player.charge_point()
        elif action == 1:  # ビーム
            player.beam_attack(opponent)
        elif action == 2:  # スペシウム光線
            player.specium_attack(opponent)
        elif action == 3:  # ガード
            player.guard()

    def _check_winner(self, player1: "Player", player2: "Player") -> int:
        """勝敗判定"""
        action1 = player1.last_action
        action2 = player2.last_action

        if action1 == "チャージ" and action2 == "チャージ":
            return 0 # ゲーム続行
        elif action1 == "チャージ" and action2 == "ビーム":
            return -1 # プレイヤー1の勝利（ビームを使用したプレイヤー2に勝ち）
        elif action1 == "チャージ" and action2 == "スペシウム光線":
            return -1 # プレイヤー2の勝利（スペシウム光線を使用したプレイヤー2に勝ち）
        elif action1 == "チャージ" and action2 == "ガード":
            return 0 # ゲーム続行
        elif action1 == "ビーム" and action2 == "ビーム":
            return 0 # ゲーム続行
        elif action1 == "ビーム" and action2 == "スペシウム光線":
            return -1 # プレイヤー2の勝利（スペシウム光線を使用したプレイヤー2に勝ち）
        elif action1 == "ビーム" and action2 == "ガード":
            return 0 # ゲーム続行
        elif action1 == "ビーム" and action2 == "チャージ":
            return 1
        elif action1 == "スペシウム光線" and action2 == "スペシウム光線":
            return 0 # ゲーム続行
        elif action1 == "スペシウム光線" and action2 == "ガード":
            return 1 # プレイヤー2の勝利（スペシウム光線を使用したプレイヤー1に勝ち)
        elif action1 == "スペシウム光線" and action2 == "ビーム":
            return 1
        elif action1 == "スペシウム光線" and action2 == "ガード":
            return 1
        elif action1 == "ガード" and action2 == "ガード":
            return 0 # ゲーム続行
        elif action1 == "ガード" and action2 == "ビーム":
            return 0
        elif action1 == "ガード" and action2 == "チャージ":
            return 0
        elif action1 == "ガード" and action2 == "スペシウム光線":
            return -1
        return 0 # デフォルト: ゲーム続行
    def render(self) -> None:
        """ゲームの状態を表示"""
        print(f"{self.player1.name}: チャージ = {self.player1.charge}, 最後の行動 = {self.player1.last_action}")
        print(f"{self.player2.name}: チャージ = {self.player2.charge}, 最後の行動 = {self.player2.last_action}")


class QLearningAgent:
    def __init__(self, actions: List[int], alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1) -> None:
        self.actions: List[int] = actions  # 行動のリスト

        self.alpha: float = alpha  # 学習率

        self.gamma: float = gamma  # 割引率
        self.epsilon: float = epsilon  # 探索率
        self.q_table: dict[Tuple[int, int], np.ndarray] = {}  # Q値テーブル

    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        """Q値を取得"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state][action]

    def update_q_value(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int]) -> None:
        """Q値を更新"""
        if next_state not in self.q_table:
            # 次の状態が初めて登場した場合、Q値を全てゼロで初期化
            self.q_table[next_state] = np.zeros(len(self.actions))
        # 次の状態における最大のQ値を取得
        best_next_action = np.max(self.q_table[next_state])
        # Q値の更新式
        current_q_value = self.get_q_value(state, action)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * best_next_action - current_q_value)
        # 更新されたQ値を保存
        self.q_table[state][action] = new_q_value

    def select_action(self, state: Tuple[int, int], show_flag = False) -> int:
        """ε-グリーディ方策による行動選択"""
        """ε-グリーディ方策とは一定確率でランダムなアクションを選択し、現在のQテーブルよりより良い行動を探索する手法"""
        charge = state[0]  # プレイヤーのチャージ数

        # 有効なアクションのリストをチャージ数に基づいて絞り込む
        available_actions = []
        # チャージ数に基づくアクション制限

        if charge >= 1:
            available_actions.append(1)  # ビーム
        if charge >= specium_cost:
            available_actions.append(2)  # スペシウム光線
        available_actions.append(0)  # チャージ（常に選択可能）
        available_actions.append(3)  # ガード（常に選択可能）
        # ε-グリーディ方策で行動選択

        if show_flag:
            valid_q_values = [self.get_q_value(state, action) for action in available_actions]
            return available_actions[np.argmax(valid_q_values)]  # 最大Q値のアクションを選択
        elif random.uniform(0, 1) < self.epsilon :
            # 探索: 有効なアクションからランダムに選択
            return random.choice(available_actions)
        else:
            # スペシウム光線を打てるなら必ず打つ
            if charge == specium_cost:
                return 2
            # 利用: Q値に基づいて有効なアクションの中から選択
            # 有効なアクションのQ値を取得
            valid_q_values = [self.get_q_value(state, action) for action in available_actions]
            return available_actions[np.argmax(valid_q_values)]  # 最大Q値のアクションを選択

    def show_agent(self):
        for my_state in range(6):
                for teki_state in range(6):
                    if self.select_action((my_state,teki_state), show_flag = True) == 0:
                        show_action = "チャージ"
                    elif self.select_action((my_state,teki_state), show_flag = True) == 1:
                        show_action = "ビーム"
                    elif self.select_action((my_state,teki_state), show_flag = True) == 2:
                        show_action = "スペシウム光線"
                    elif self.select_action((my_state,teki_state), show_flag = True) == 3:
                        show_action = "ガード"
                    print(f"自分のチャージが{my_state}, 相手のチャージが{teki_state} : {show_action}")

def cal_reward(state : Tuple[int,int], win : int) -> float:
    # 勝利報酬
    win_reward = 0
    if win == 1:
        win_reward = 100000
    elif win == -1:
        win_reward = -100000
    return win_reward


def train_agent(episodes: int = 1000) -> Tuple[Any, Any]:
    env = CCGameEnv()  # 環境の初期化

    agent1 = QLearningAgent(actions=[0, 1, 2, 3])  # プレイヤー1のエージェント
    agent2 = QLearningAgent(actions=[0, 1, 2, 3])  # プレイヤー2のエージェント
    # episode log
    episode_list = []
    end_turn_list = []
    winner_list = []
    # turn log
    episode_list_for_turn_log = []
    turn_list = []
    before_player_one_charage_list = []
    before_player_two_charage_list = []
    after_player_one_charage_list = []
    after_player_two_charage_list = []
    player_one_command_list = []
    player_two_command_list = []
    result_list = []
    # Q tables log
    total_reward_list = []
    after_q_tables_list = []
    before_q_tables_list = []

    # Qテーブルの初期化
    for episode in range(episodes):
        state = env.reset()  # ゲームの初期状態
        done = False
        total_reward = 0
        turn = 0

        while not done:
            # commandの決定
            turn += 1
            action1 = agent1.select_action(state)  # プレイヤー1の行動
            action2 = agent2.select_action(state)  # プレイヤー2の行動
            # actionの実行前に保存できるログを記録
            episode_list_for_turn_log.append(episode)
            turn_list.append(turn)
            before_player_one_charage_list.append(state[0])
            before_player_two_charage_list.append(state[1])
            player_one_command_list.append(action1)
            player_two_command_list.append(action2)
            before_q_tables_list.append(agent1.q_table)

            # actionを実行して状態を更新
            next_state, win, done = env.step(action1, action2)
            # 状態と報酬の更新
            state = next_state
            total_reward += cal_reward(state,win)
            # Q値の更新
            agent1.update_q_value(state, action1, total_reward, next_state)
            agent2.update_q_value(state, action2, -total_reward, next_state)
            # actionの実行後のログを記録
            total_reward_list.append(total_reward)
            after_player_one_charage_list.append(state[0])
            after_player_two_charage_list.append(state[1])
            after_q_tables_list.append(agent1.q_table)
            result_list.append(win)
        if episode % 1000 == 0:
            print(f"Episode {episode}, Total reward: {total_reward}")
        

        # episode 終了時にepisode のログを記録する
        episode_list.append(episode)
        end_turn_list.append(turn)
        winner_list.append(win)
    
    episode_log = pd.DataFrame({
        "episode_id": episode_list,
        "end_turn": end_turn_list,
        "winner": winner_list,
    })
    turn_log = pd.DataFrame({
        "episode_id":  episode_list_for_turn_log,
        "turn": turn_list,
        "before_player_one_charage": before_player_one_charage_list,
        "before_player_two_charage": before_player_two_charage_list,
        "after_player_one_charage": after_player_one_charage_list,
        "after_player_two_charage": after_player_two_charage_list,
        "player_one_command": player_one_command_list,
        "player_two_command": player_two_command_list,
        "result": result_list,
    })
    q_tables_log = pd.DataFrame({
        "episode_id": episode_list_for_turn_log,
        "turn": turn_list,
        "total_reward": total_reward_list,
        "Q-tables": after_q_tables_list,
    })

    return agent1, agent2, episode_log, turn_log, q_tables_log

def buttle_random_agent(agent, episodes: int = 1000) -> int:
    env = CCGameEnv()  # 環境の初期化
    random_agent = QLearningAgent(actions=[0, 1, 2, 3])

    total_win = 0
    for episode in range(episodes):
        state = env.reset()  # ゲームの初期状態
        done = False
        while not done:
            action1 = agent.select_action(state)  # プレイヤー1の行動
            action2 = random_agent.select_action(state)  # プレイヤー2の行動
            next_state, win, done = env.step(action1, action2)
            state = next_state
        if episode % 100 == 0 and episode > 0:
            print(f"Episode {episode}, Total win: {total_win}, win ratio: {total_win / episode}")
        if win == 1:
            total_win += 1
    return total_win

agent1, agent2, episode_list, turn_list, q_tables_log = train_agent(episodes=10000)