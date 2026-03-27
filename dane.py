import pandas as pd
from binance.client import Client
import datetime
import time
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Ustawienia
SYMBOL = "BTCUSDT"  # Para tradingowa
INTERVAL = Client.KLINE_INTERVAL_15MINUTE  # Interwał 15-minutowy
START_DATE = "1 Mar, 2020"  # Początek danych
END_DATE = "25 Mar, 2025"  # Koniec danych (dzisiejsza data)
OUTPUT_FILE = "btc_usd_2020_2025_15mv2.csv"

# Inicjalizacja klienta Binance (bez klucza API, bo to publiczne dane)
client = Client("", "")  # Puste klucze, bo nie potrzebujemy autoryzacji do klines

def fetch_historical_data(symbol, interval, start_str, end_str):
    """Pobiera dane historyczne z Binance API w porcjach"""
    logger.info(f"Pobieranie danych dla {symbol}, interwał: {interval}, od {start_str} do {end_str}")
    
    # Konwersja dat na timestamp w milisekundach
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)
    limit = 1000  # Maksymalna liczba świec na zapytanie
    all_klines = []
    current_ts = start_ts

    while current_ts < end_ts:
        try:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(current_ts),
                end_str=str(min(current_ts + limit * 15 * 60 * 1000, end_ts)),  # 15 min * limit
                limit=limit
            )
            if not klines:
                break
            all_klines.extend(klines)
            current_ts = int(klines[-1][0]) + 1  # Następny timestamp
            logger.info(f"Pobrano {len(klines)} świec, timestamp: {datetime.datetime.utcfromtimestamp(current_ts / 1000)}")
            time.sleep(0.1)  # Limit zapytań Binance
        except Exception as e:
            logger.error(f"Błąd: {e}")
            time.sleep(5)  # Czekaj przy błędzie
            continue

    return all_klines

def process_data(klines):
    """Przetwarza dane do DataFrame"""
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close"]]  # Podstawowe kolumny OHLC
    df = df.astype({"open": float, "high": float, "low": float, "close": float})

    # Dodanie Kill Zones
   # kill_zones = [(13, 16), (2, 5), (20, 23)]  # Godziny UTC
   # df["Kill_Zone"] = 0
  #  for start, end in kill_zones:
        # Ustawienie 1 dla godzin w Kill Zone
   #     df.loc[df["timestamp"].dt.hour.between(start, end-1), "Kill_Zone"] = 1
    
 #   logger.info("Dane przetworzone - Kill Zones dodane, wierszy: %d", len(df))
    return df

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=period).mean()
    return df

def detect_order_blocks(df):
    df["OB_bullish"] = 0
    df["OB_bearish"] = 0
    for i in range(2, len(df)):
        if (df["close"].iloc[i-2] < df["open"].iloc[i-2] and 
            df["close"].iloc[i-1] > df["high"].iloc[i-2] and 
            df["close"].iloc[i] > df["high"].iloc[i-1]):
            df.loc[df.index[i-1], "OB_bullish"] = 1
        elif (df["close"].iloc[i-2] > df["open"].iloc[i-2] and 
              df["close"].iloc[i-1] < df["low"].iloc[i-2] and 
              df["close"].iloc[i] < df["low"].iloc[i-1]):
            df.loc[df.index[i-1], "OB_bearish"] = 1
    return df

def detect_fvg(df):
    df["FVG_start"] = 0.0
    df["FVG_end"] = 0.0
    df["FVG_direction"] = "null"
    for i in range(2, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i-2]:  # Luka w górę
            df.loc[df.index[i-1], "FVG_start"] = df["high"].iloc[i-2]
            df.loc[df.index[i-1], "FVG_end"] = df["low"].iloc[i]
            df.loc[df.index[i-1], "FVG_direction"] = "up"
        elif df["high"].iloc[i] < df["low"].iloc[i-2]:  # Luka w dół
            df.loc[df.index[i-1], "FVG_start"] = df["low"].iloc[i-2]
            df.loc[df.index[i-1], "FVG_end"] = df["high"].iloc[i]
            df.loc[df.index[i-1], "FVG_direction"] = "down"
    return df

def add_kill_zones(df):
    kill_zones = [(13, 16), (2, 5), (20, 23)]  # Godziny UTC
    df["Kill_Zone"] = 0
    for start, end in kill_zones:
        df.loc[df["timestamp"].dt.hour.between(start, end-1), "Kill_Zone"] = 1
    return df

def save_to_csv(df, filename):
    """Zapisuje dane do pliku CSV"""
    df.to_csv(filename, index=False)
    logger.info(f"Dane zapisane do {filename}, wierszy: {len(df)}")

def main():
    # Pobierz dane
    klines = fetch_historical_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    # Przetwórz dane
    df = process_data(klines)
    df = calculate_atr(df)
    df = detect_order_blocks(df)
    df = detect_fvg(df)
    df = add_kill_zones(df)
    # Zapisz do CSV
    save_to_csv(df, OUTPUT_FILE)

if __name__ == "__main__":
    main()