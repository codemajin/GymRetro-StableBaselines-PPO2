# -*- coding: utf-8 -*-


from typing import Tuple
import numpy as np
import gym
import retro
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from baselines.common.retro_wrappers import (
    StochasticFrameSkip,
    Downsample,
    Rgb2gray,
    FrameStack,
    ScaledFloatFrame
)


class AirstrikerDiscretizer(gym.ActionWrapper):
    """行動空間を離散空間に変換する機能を持つ
    """

    def __init__(self, env: gym.Env) -> None:
        """初期化する

        Args:
            env (gym.Env): 環境のインスタンス
        """
        super(AirstrikerDiscretizer, self).__init__(env)
        buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
                   'C', 'Y', 'X', 'Z']
        actions = [['LEFT'], ['RIGHT'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, index: np.int64) -> np.ndarray:
        """行動を取得する

        Args:
            index (np.int64): インデックス

        Returns:
            np.ndarray: 行動
        """
        return self._actions[index].copy()


class CustomRewardAndDoneEnv(gym.Wrapper):
    """報酬とエピソード完了の変更
    """
    def __init__(self, env: gym.Env) -> None:
        """初期化する

        Args:
            env (gym.Env): 環境のインスタンス
        """
        super(CustomRewardAndDoneEnv, self).__init__(env)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """行動を1ステップ実行する。

        Args:
            action ([type]): 行動

        Returns:
            Tuple[np.ndarray, float, bool, dict]: 状態, 報酬, 完了フラグ, デバック情報
        """
        state, reward, done, info = self.env.step(action)

        # 報酬の変更
        reward /= 20

        # エピソード完了の変更
        if info['gameover'] == 1:
            done = True

        return state, reward, done, info


def make_environment(log_dir: str, game: str = 'Airstriker-Genesis',
                     state: str = 'Level1', seeds: int = 0) -> DummyVecEnv:
    """強化学習の環境を作成する。

    Args:
        log_dir (str): ログの出力先ディレクトリ
        game (str): ゲーム名 (デフォルトは 'Airstriker-Genesis')
        state (str): ゲームの初期状態 (デフォルトは 'Level1')
        seeds (int): 乱数の種 (デフォルトは 0)

    Returns:
        DummyVecEnv: 強化学習の環境
    """
    env = retro.make(game=game, state=state)

    # 行動空間を離散空間に変換
    env = AirstrikerDiscretizer(env)

    # 報酬とエピソード完了の変更
    env = CustomRewardAndDoneEnv(env)

    # 4フレームごとに行動を選択し、4フレーム連続で同じ行動をとる
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)

    # ゲーム画面のサイズを1/2に縮小する
    env = Downsample(env, 2)

    # ゲーム画面をグレースケールに変換
    env = Rgb2gray(env)

    # 直近4フレーム分の画面イメージを環境の状態とする
    env = FrameStack(env, 4)

    # 環境の状態の正規化 (ゲーム画面の画素値を0.0〜1.0へ正規化)
    env = ScaledFloatFrame(env)

    # 環境のログを出力する
    env = Monitor(env, log_dir, allow_early_resets=True)

    # シードの設定
    env.seed(seeds)

    return DummyVecEnv([lambda: env])
