# -*- coding: utf-8 -*-


import os
import numpy as np
import datetime
import pytz
from stable_baselines.results_plotter import load_results, ts2xy


class TrainHook:
    """学習状況を監視するためのコールバックメソッドを有する
    """

    def __init__(self, log_dir: str, saved_file_name: str) -> None:
        """初期化する

        Args:
            log_dir (str): ログの保存先
            saved_file_name (str): モデルのファイル名
        """
        self.__log_dir = log_dir
        self.__saved_file_name = saved_file_name
        self.__best_mean_reward = -np.inf
        self.__nupdates = 0

    def __call__(self, _locals: dict, _globals: dict) -> bool:
        """強化学習時のフックメソッド

        Args:
            _locals (dict): ローカル変数の辞書
            _globals (dict): グローバル変数の辞書 (未使用)

        Returns:
            bool: 学習を継続するかどうか (常にTrueを返す)
        """
        # 10更新毎
        if (self.__nupdates + 1) % 10 == 0:
            # 平均エピソード長、平均報酬の取得
            x, y = ts2xy(load_results(self.__log_dir), 'timesteps')
            if len(y) > 0:
                # 最近10件の平均報酬
                mean_reward = np.mean(y[-10:])

                # 平均報酬がベスト報酬以上の時はモデルを保存
                model_updated = self.__update_model(_locals, mean_reward)

                # ログ
                self.__debug_log(mean_reward, model_updated)

        self.__nupdates += 1
        return True

    def __update_model(self, _locals: dict, mean_reward: float) -> bool:
        """平均報酬がベスト報酬以上の時はモデルを保存する

        Args:
            _locals (dict): ローカル変数の辞書
            mean_reward (float): 直近の平均報酬

        Returns:
            bool: モデルを保存したかどうか
        """
        need_update = mean_reward > self.__best_mean_reward

        if need_update:
            self.__best_mean_reward = mean_reward

            file_name = os.path.join(self.__log_dir, f"{self.__saved_file_name}_{self.__nupdates}")
            _locals['self'].model.save(file_name)

        return need_update

    def __debug_log(self, mean_reward: float, model_updated: bool) -> None:
        """デバッグログを表示する

        Args:
            mean_reward (float): 直近の平均報酬
            model_updated (bool): モデルを保存したかどうか
        """
        print('time: {}, nupdates: {}, mean: {:.2f}, best_mean: {:.2f}, model_updated: {}'.format(
              datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
              self.__nupdates, mean_reward, self.__best_mean_reward, model_updated))
