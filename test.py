def train_agent(episodes: int = 1000) -> Tuple[Any, Any]:
    env = CCGameEnv()  # 環境の初期化

    agent1 = QLearningAgent(actions=[0, 1, 2, 3])  # プレイヤー1のエージェント
    agent2 = QLearningAgent(actions=[0, 1, 2, 3])  # プレイヤー2のエージェント
    random_agent = QLearningAgent(actions=[0,1,2,3])
    for episode in range(episodes):
        state = env.reset()  # ゲームの初期状態
        done = False
        total_reward = 0
            action1 = agent1.select_action(state, episode, episodes)  # プレイヤー1の行動
            action2 = agent2.select_action(state, episode, episodes)  # プレイヤー2の行動
            next_state, win, done = env.step(action1, action2)  # アクションを実行
            # Q値の更新
            agent1.update_q_value(state, action1, win, next_state)

            agent2.update_q_value(state, action2, -win, next_state)
            state = next_state
            total_reward += cal_reward(state,win)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total reward: {total_reward}")
    return agent1, agent2