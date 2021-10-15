# GymRetro-StableBaselines-PPO2

[Gym Retro](https://openai.com/blog/gym-retro/) の環境を使用して、強化学習パッケージである [Stable Baselines](https://github.com/hill-a/stable-baselines) の [PPO2 (Proximal Policy Optimization)](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) アルゴリズムを試す。

<img src="https://user-images.githubusercontent.com/20922926/137328743-b6ff5fbb-3a90-4959-a0b3-6abf4b2df10b.gif" alt="50" style="zoom:200%;" />

## 環境構築

**Ubuntu 20.04** 上に導入した Anaconda環境で動作確認済み (Windows や mac では未確認)。環境構築手順は次の通り。

1. まず、必要なシステムパッケージをインストールする。
    ```bash
    $ sudo apt install python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
    ```

2. 次に、Anacondaの仮想環境を作成する。Pythonのバージョンは、**3.7** を指定すること。
    ```bash
    $ conda create -n <仮想環境名> python=3.7 anaconda
    $ conda activate <仮想環境名>
    ```

3. 必要なパッケージをインストールする。

    ```bash
    $ conda install tensorflow-gpu==1.14.0
    $ pip install gym
    $ pip install gym-retro
    $ pip install stable-baselines[mpi]
    ```
4. [OpenAI Baselines](https://github.com/openai/baselines)の導入が必要だが、先に[mujoco-py](https://github.com/openai/mujoco-py)をインストールする（リンク先の手順を参照）。

5. [OpenAI Baselines](https://github.com/openai/baselines)の導入が必要だが、先に[mujoco-py](https://github.com/openai/mujoco-py)をインストールする。

    ```bash
    $ git clone https://github.com/openai/baselines.git
    $ cd baselines
    $ pip install -e .
    ```
## 使い方

学習用と、学習済モデルの実行それぞれにスクリプトファイルを用意している。使い方は、次の通り。

### 学習方法

学習用のスクリプト`train.py`を引数なし(デフォルト設定)で実行できる。必要に応じて引数を指定すること。

```bash
$ python train.py [--logging_dir=<ログの出力先>] [--saved_file_name=<学習済みモデルのファイル名>] [--total_timesteps=<学習の総ステップ数>] [--seeds=<乱数のシード>]
```

コマンドライン引数の詳細は次の通り。

```bash
$ python train.py --help
       USAGE: train.py [flags]
flags:

train.py:
  --logging_dir: ログの出力先ディレクトリを指定する.
    (default: './logs')
  --saved_file_name: 学習済みモデルの保存ファイル名で, 拡張子は指定しない.
    (default: 'trained_model')
  --seeds: 乱数のシードを指定する.
    (default: '0')
    (an integer)
  --total_timesteps: 学習の総ステップ数を指定する
    (default: '128000')
    (an integer)

Try --helpfull to get a list of all flags.
```

### 動作確認

学習済みモデルの動作確認は、`play.py`で実行できる。適宜、コマンドライン引数をしていること。

```bash
$ python play.py [--logging_dir=<ログの出力先>] [--saved_file_name=<学習済みモデルのファイル名>] [--movie_file_name=<ゲームの実行画面を保存する際の動画ファイル名>] [--n_episodes=<ゲームの実行回数>]
```

コマンドライン引数の詳細は次の通り。

```bash
$ python play.py --help
       USAGE: play.py [flags]
flags:

play.py:
  --logging_dir: ログの出力先ディレクトリを指定する.
    (default: './logs')
  --movie_file_name: モデルが実行したゲーム画面を保存する動画ファイル名
    (default: 'play_movie.mp4')
  --n_episodes: ゲームの実行回数を指定する
    (default: '5')
    (an integer)
  --saved_file_name: 学習済みモデルの保存ファイル名で, 拡張子は指定しない.
    (default: './logs/trained_model')

Try --helpfull to get a list of all flags.
```