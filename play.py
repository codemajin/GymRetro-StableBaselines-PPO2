# -*- coding: utf-8 -*-


import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=Warning)

import logging
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

import time
import cv2
from typing import List
from absl import app, flags
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from environment import make_environment


# コマンドライン引数
flags.DEFINE_string('logging_dir', './logs', 'ログの出力先ディレクトリを指定する.')
flags.DEFINE_string('saved_file_name', 'trained_model', '学習済みモデルの保存ファイル名で, 拡張子は指定しない.')
flags.DEFINE_string('movie_file_name', 'ai_play.mp4', 'モデルが実行したゲーム画面を保存する動画ファイル名')
flags.DEFINE_integer('n_episodes', 5, 'ゲームの実行回数を指定する')


class MovieWriter:
    """動画を保存する機能を有する
    """

    def __init__(self, movie_file_name: str) -> None:
        """インスタンスを初期化する

        Args:
            movie_file_name (str): 動画ファイル名
        """
        self.__movie_file_name = movie_file_name
        self.__writer = None

    def __del__(self) -> None:
        """インスタンスを破棄する
        """
        if self.__writer is not None:
            self.__writer.release()

    def __call__(self, env: DummyVecEnv) -> None:
        """ゲームの実行画面を保存する

        Args:
            env (DummyVecEnv): 環境
        """
        rgb = env.render('rgb_array')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.__writer is None:
            width, height = rgb.shape[:2]
            self.__writer = self.__build_writer(width, height)

        self.__writer.write(rgb)

    def __build_writer(self, width: int, height: int) -> cv2.VideoWriter:
        """動画保存のオブジェクトを生成する

        Args:
            width (int): 動画画面の幅
            height (int): 動画画面の高さ

        Returns:
            cv2.VideoWriter: 動画保存のオブジェクト
        """
        fps = 24
        size = (height, width)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(self.__movie_file_name, fourcc, fps, size)


def play(_argv: List[str]) -> None:
    """学習済みモデルでゲームを実行する

    Args:
        _argv (List[str]): コマンドライン引数
    """
    # 環境の作成
    env = make_environment(log_dir=flags.FLAGS.logging_dir)

    # 学習済みのモデルを読み込み
    model = PPO2.load(flags.FLAGS.saved_file_name, env=env, verbose=0)

    # 動画保存用のインスタンス
    movie = MovieWriter(flags.FLAGS.movie_file_name)

    # 初期状態に設定
    n_play_count = 0
    total_reward = 0.0
    state = env.reset()

    # ゲームを実行する
    while n_play_count < flags.FLAGS.n_episodes:
        # 環境の描画
        env.render()

        # 動画保存
        movie(env)

        # スリープ
        time.sleep(1/60)

        # モデルの推論
        action, _ = model.predict(state)

        # 1ステップ実行
        state, reward, done, _ = env.step(action)
        total_reward += reward[0]

        # エピソード完了
        if done:
            print('reward:', total_reward)
            state = env.reset()
            total_reward = 0
            n_play_count += 1


if __name__ == '__main__':
    print('')
    app.run(play)
