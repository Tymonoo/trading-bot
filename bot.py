# v11 - Trading Bot - Część 1: Importy i konfiguracja
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Stałe
LEVERAGE = 10
INITIAL_BALANCE = 10000
KILL_ZONES = [
    (13, 16),  # 13:00-16:00 UTC
    (2, 5),    # 2:00-5:00 UTC
    (20, 23)   # 20:00-23:00 UTC
]
ATR_PERIOD = 14
CONTEXT_DAYS = 1  # 1 dzień kontekstu (96 świec 15M)

# Funkcja sprawdzająca Kill Zone
def is_in_kill_zone(timestamp):
    dt = pd.to_datetime(timestamp)
    hour = dt.hour
    for start, end in KILL_ZONES:
        if start <= hour < end:
            return 1
    return 0

# Ładowanie danych
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Kill_Zone"] = df["timestamp"].apply(is_in_kill_zone)
    # Konwersja typów i wypełnienie NaN przed przekazaniem
    required_cols = ["open", "high", "low", "close", "ATR", "OB_bullish", "OB_bearish", "FVG_start", "FVG_end", "Kill_Zone"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny {col} w danych")
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)
    if "FVG_direction" not in df.columns:
        df["FVG_direction"] = "none"
    df["FVG_direction"] = df["FVG_direction"].fillna("none").astype(str)
    return df

# v11 - Trading Bot  - Część 2: Środowisko TradingEnv
class TradingEnv(gym.Env):
    def __init__(self, data, leverage=LEVERAGE, initial_balance=INITIAL_BALANCE):
        super(TradingEnv, self).__init__()
        self.data = data
        self.leverage = leverage
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0: brak, 1: long, -1: short
        self.position_size = 0
        self.entry_price = 0
        self.step_idx = 0
        self.logs = []

        # Przestrzeń akcji: 0 (zamknij), 1 (long), 2 (short), 3 (czekaj)
        self.action_space = spaces.Discrete(4)

        # Przestrzeń obserwacji: OHLC, ATR, OB, FVG, Kill_Zone + kontekst 1D
        state_size = 10 + 96  # 10 cech bieżącej świecy + 96 zamknięć kontekstu
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

        # Weryfikacja danych na starcie
      #  required_cols = ["open", "high", "low", "close", "ATR", "OB_bullish", "OB_bearish", "FVG_start", "FVG_end", "Kill_Zone"]
      #  for col in required_cols:
      #      if col not in self.data.columns:
       #         raise ValueError(f"Brak kolumny {col} w danych")
       #     self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.step_idx = 0
        self.logs = []
        logger.info("Reset środowiska - saldo początkowe: %.2f USD", self.balance)
        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        row = self.data.iloc[self.step_idx]
        # Pobierz kontekst, upewnij się, że zawsze mamy 96 elementów
        context_start = max(0, self.step_idx - 96)
        context = self.data.iloc[context_start:self.step_idx]["close"].values.astype(np.float32)
        if len(context) < 96:
            padding_size = 96 - len(context)
            context = np.pad(context, (padding_size,0), mode='constant', constant_values=0.0)
        context = context.astype(np.float32)
        
        state = np.array([
            row["open"], row["high"], row["low"], row["close"], row["ATR"],
            row["OB_bullish"], row["OB_bearish"], row["FVG_start"], row["FVG_end"],
            row["Kill_Zone"]
        ], dtype=np.float32)
        state = np.concatenate([state, context]).astype(np.float32) / 10000  # Normalizacja
        if len(state) != 106:
            raise ValueError(f"Nieprawidłowy rozmiar obserwacji: {len(state)}, oczekiwano 106")
        return state

    def step(self, action, training=True):
        if self.step_idx >= len(self.data):
            return None, 0, True, False, {}  # Zakończ, jeśli poza danymi
        
        row = self.data.iloc[self.step_idx]
        atr = row["ATR"]
        sl = -0.75 * atr * self.leverage
        tp = 2 * atr * self.leverage
        reward = 0
        reason = "Nieokreślony"
        entry_price_log = self.entry_price if self.position != 0 else 0.0
        exit_price_log = None
        position_size_log = self.position_size
        btc_price_log = row["close"]  # Dodajemy cenę BTC
        timestamp_log = row["timestamp"]  # Dodajemy timestamp
        # Debugowanie
        print(f"Krok {self.step_idx}: Akcja: {action}, Kill_Zone: {row['Kill_Zone']}, "
             f"OB_bullish: {row['OB_bullish']}, FVG_direction: {row['FVG_direction']}")

        if self.position == 0:
            if action == 1 and row["Kill_Zone"] == 1 and (row["OB_bullish"] == 1 or row["FVG_direction"] == "up"):
                self.position = 1
                self.entry_price = row["close"]
                self.position_size = self.balance * 0.1
                entry_price_log = self.entry_price
                position_size_log = self.position_size
                reason = "Long: Bullish OB w Kill Zone" if row["OB_bullish"] else "Long: FVG up w Kill Zone"
                reward = 0.1
            elif action == 2 and row["Kill_Zone"] == 1 and (row["OB_bearish"] == 1 or row["FVG_direction"] == "down"):
                self.position = -1
                self.entry_price = row["close"]
                self.position_size = self.balance * 0.1
                entry_price_log = self.entry_price
                position_size_log = self.position_size
                reason = "Short: Bearish OB w Kill Zone" if row["OB_bearish"] else "Short: FVG down w Kill Zone"
                reward = 0.1
            elif action == 3:
                reward = 0.1 if row["Kill_Zone"] == 0 else -0.5
                reason = "Czekaj: Poza Kill Zone" if row["Kill_Zone"] == 0 else "Czekaj: Brak OB/FVG w Kill Zone"
            else:
                reward = -0.1  # Kara za błędną akcję
                reason = "Błąd: Brak sygnału dla akcji"

        elif self.position != 0:
            price_diff = (row["close"] - self.entry_price) * self.position
            profit = (price_diff/self.entry_price) * self.position_size * self.leverage
            if action == 0 or profit >= tp or profit <= sl:
                self.balance += profit
                exit_price_log = row["close"]
                if profit >= tp:
                    reward = 20
                    reason = f"Zamknięcie: TP osiągnięte (+{profit:.2f} USD)"
                elif profit <= sl:
                    reward = -15
                    reason = f"Zamknięcie: SL osiągnięte ({profit:.2f} USD)"
                else:
                    reward = 0
                    reason = "Zamknięcie: Brak sygnału"
                self.position = 0
                self.position_size = 0
                position_size_log = 0
                entry_price_log = 0.0
                logger.info("Zamknięcie pozycji - Saldo: %.2f USD, Zysk/Strata: %.2f USD", self.balance, profit)

        self.logs.append([self.step_idx, action, self.balance, reward, reason, entry_price_log, exit_price_log, position_size_log, btc_price_log, timestamp_log])
        logger.info("Krok %d: Akcja: %d, Saldo: %.2f, Nagroda: %.2f, Powód: %s, Cena_wejścia: %.2f, Cena_wyjścia: %s, Wielkość_pozycji: %.2f, BTC_price: %.2f, Timestamp: %s", 
                    self.step_idx, action, self.balance, reward, reason, entry_price_log, str(exit_price_log), position_size_log, btc_price_log, str(timestamp_log))

        self.step_idx += 1
        done = self.step_idx >= len(self.data) or self.balance < 10
        truncated = False
        if done and training:
            # Reset środowiska, jeśli dane się skończyły lub saldo za niskie
            obs, _ = self.reset()
            logger.info("Środowisko zresetowane po zakończeniu epizodu")
        elif done:
            obs = None
        else:
            obs = self._get_observation()
        return obs, reward, done, truncated,  {}

    def render(self, mode="human"):
        if self.step_idx > 0:
            last_log = self.logs[-1]
            print(f"Krok: {last_log[0]}, Akcja: {last_log[1]}, Saldo: {last_log[2]:.2f}, Nagroda: {last_log[3]:.2f}, Powód: {last_log[4]},, Cena_wejścia: {last_log[5]:.2f}, Cena_wyjścia: {last_log[6] if last_log[6] is not None else 'Brak'}, Cena_BTC: {last_log[8]:.2f}, Timestamp: {last_log[9]}")
            # v11 - Trading Bot z (po tuningu) - Część 3: Trening i testy
def train_and_test():
    # Ładowanie danych
    data = load_data("btc_usd_2020_2025_15m.csv")
    train_data = data[data["timestamp"] < "2024-03-01"]
    test_data = data[data["timestamp"] >= "2024-03-01"]
    
    logger.info("Dane załadowane - Trening: %d świec, Testy: %d świec", len(train_data), len(test_data))

    # Inicjalizacja środowiska treningowego
    env = TradingEnv(train_data)
    check_env(env)  # Weryfikacja zgodności z SB3
    env = DummyVecEnv([lambda: env])  # Wektorizacja dla PPO

    # Tworzenie i trening modelu
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    logger.info("Rozpoczęcie treningu - 5M kroków")
    model.learn(total_timesteps=5_000_000, progress_bar=True)
    model.save("v12_tuned_model")
    logger.info("Trening zakończony - Model zapisany jako v12_tuned_model")

    # Testowanie
    test_env = TradingEnv(test_data)
    obs, _ = test_env.reset()
    logger.info("Rozpoczęcie testów - Saldo początkowe: %.2f USD", test_env.balance)

    for step in range(len(test_data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action, training=False)
        test_env.render()
        if done or truncated:
            logger.info("Testy zakończone przedwcześnie - Saldo: %.2f USD", test_env.balance)
            break

    # Wyniki
    final_balance = test_env.balance
    logs_df = pd.DataFrame(test_env.logs, columns=["Krok", "Akcja", "Saldo", "Nagroda", "Powód", "Cena_wejścia", "Cena_wyjścia", "Wielkość pozycji"])
    logs_df.to_csv("v11_tuned_logs.csv", index=False)

    logger.info("Testy zakończone - Saldo końcowe: %.2f USD", final_balance)
    print(f"Saldo końcowe: {final_balance:.2f} USD")
    print("Ostatnie 5 wpisów w logach:")
    print(logs_df.tail())


def analyze_results(logs_file="v12_tuned_logs.csv"):
    """Analiza wyników testów"""
    logs_df = pd.read_csv(logs_file)
    
    # Kluczowe metryki
    initial_balance = 10000

    if logs_df.empty:
        logger.warning("Logi są puste - brak danych do analizy")
        final_balance = initial_balance
        total_profit = 0
        profit_percent = 0
        win_rate = 0
        max_drawdown = 0
        mdd_percent = 0
        num_trades = 0
    else:
        final_balance = logs_df["Saldo"].iloc[-1]
        total_profit = final_balance - initial_balance
        profit_percent = (total_profit / initial_balance) * 100

        logs_df["Powód"] = logs_df["Powód"].fillna("Nieokreślony")
        trades = logs_df[logs_df["Powód"].str.contains("Zamknięcie")]
        winning_trades = trades[trades["Nagroda"] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
    
        max_drawdown = 0
        peak = initial_balance
        for balance in logs_df["Saldo"]:
            peak = max(peak, balance)
            drawdown = peak - balance
            max_drawdown = max(max_drawdown, drawdown)
        mdd_percent = (max_drawdown / peak) * 100 if peak > 0 else 0
        num_trades = len(trades)

    # Wyświetlanie wyników
    logger.info("Analiza wyników:")
    print(f"Saldo początkowe: {initial_balance:.2f} USD")
    print(f"Saldo końcowe: {final_balance:.2f} USD")
    print(f"Zysk całkowity: {total_profit:.2f} USD (+{profit_percent:.2f}%)")
    print(f"Liczba transakcji: {num_trades}")
    print(f"% zyskownych pozycji: {win_rate:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f} USD ({mdd_percent:.2f}%)")

    return {
        "final_balance": final_balance,
        "profit_percent": profit_percent,
        "win_rate": win_rate,
        "max_drawdown": mdd_percent,
        "num_trades": num_trades
    }

def validate_data(data):
    """Weryfikacja poprawności danych"""
    required_columns = ["timestamp", "open", "high", "low", "close", "ATR", 
                       "OB_bullish", "OB_bearish", "FVG_start", "FVG_end", "FVG_direction"]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        logger.error("Brakujące kolumny w danych: %s", missing)
        raise ValueError(f"Brakujące kolumny: {missing}")
    
    if data["timestamp"].isnull().any():
        logger.error("Puste wartości w kolumnie timestamp")
        raise ValueError("Puste wartości w timestamp")
    
    logger.info("Dane zweryfikowane - OK")
    return True


if __name__ == "__main__":
    # Ładowanie i weryfikacja danych
    data = load_data("btc_usd_2020_2025_15m.csv")
    validate_data(data)
    
    # Podział danych
    train_data = data[data["timestamp"] < "2024-03-01"]
    test_data = data[data["timestamp"] >= "2024-03-01"]
    logger.info("Dane załadowane - Trening: %d świec, Testy: %d świec", len(train_data), len(test_data))

    # Trening
    env = TradingEnv(train_data)
    check_env(env)
    env = DummyVecEnv([lambda: env])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    logger.info("Rozpoczęcie treningu - 5M kroków")
    model.learn(total_timesteps=5_000_000, progress_bar=True)
    model.save("v12_tuned_model")
    logger.info("Trening zakończony - Model zapisany jako v12_tuned_model")

    # Testowanie
    test_env = TradingEnv(test_data)
    obs, _ = test_env.reset()
    logger.info("Rozpoczęcie testów - Saldo początkowe: %.2f USD", test_env.balance)
    for step in range(len(test_data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action, training=False)
        test_env.render()
        if done or truncated:
            logger.info("Testy zakończone - Saldo: %.2f USD", test_env.balance)
            break
        if obs is None:
            logger.warning("Obserwacja None - koniec danych")
            break

    # Wyniki
    final_balance = test_env.balance
    logs_df = pd.DataFrame(test_env.logs, columns=["Krok", "Akcja", "Saldo", "Nagroda", "Powód", 
                                                   "Cena_wejścia", "Cena_wyjścia", "Wielkość pozycji", 
                                                   "Cena_BTC", "Timestamp"])
    logs_df.to_csv("v12_tuned_logs.csv", index=False)
    print(f"Saldo końcowe: {final_balance:.2f} USD")
    print("Ostatnie 5 wpisów w logach:")
    print(logs_df.tail())
    analyze_results()