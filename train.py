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

from absl import app, flags
from typing import List
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from environment import make_environment
from train_hook import TrainHook


# コマンドライン引数
flags.DEFINE_string('logging_dir', './logs', 'ログの出力先ディレクトリを指定する.')
flags.DEFINE_string('saved_file_name', 'trained_model', '学習済みモデルの保存ファイル名で, 拡張子は指定しない.')
flags.DEFINE_integer('total_timesteps', 128000, '学習の総ステップ数を指定する')
flags.DEFINE_integer('seeds', 0, '乱数のシードを指定する.')


def train(_argv: List[str]) -> None:
    """強化学習を実行する

    Args:
        _argv (List[str]): コマンドライン引数
    """
    # ログのディレクトリ作成
    os.makedirs(flags.FLAGS.logging_dir, exist_ok=True)

    # 環境の作成
    env = make_environment(log_dir=flags.FLAGS.logging_dir, seeds=flags.FLAGS.seeds)

    # シードの指定
    set_global_seeds(flags.FLAGS.seeds)

    # モデルの生成
    model = PPO2('CnnPolicy', env, verbose=0)

    # モデルの学習
    model.learn(total_timesteps=flags.FLAGS.total_timesteps,
                callback=TrainHook(flags.FLAGS.logging_dir, flags.FLAGS.saved_file_name))

    # モデルの保存
    model.save(flags.FLAGS.saved_file_name)


if __name__ == '__main__':
    app.run(train)
