#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH_PROB_V2_SIGNAL.py

V2 del bot ETHUSDT Perpetual:
- Datos PERP multi-fuente: OKX / KuCoin / Bybit / Binance, no spot.
- Multi-timeframe: 5m / 15m / 1h / 4h.
- Proyecciones 1h y 6h mediante analogías históricas continuas (KNN simple).
- Tres estados: ALCISTA / NEUTRAL / BAJISTA.
- Filtro NO TRADE cuando no existe ventaja estadística suficiente.
- Entrada por ZONA, no por precio exacto.
- SL estructural + ATR de 15m (evita stops dentro del ruido de 1m).
- TP1 / TP2 / TP3 por múltiplos de R.
- Ruptura por cierre sobre/baixo estructura + spike de volumen.
- Usa solo velas cerradas.
- Telegram mediante GitHub Secrets.
"""

import os
import time
import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import certifi
import numpy as np
import pandas as pd
import pytz
import requests

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ETH_PROB_V2")

VERSION = "V2.4-DIRECTIONAL-2026-09-06"


# ============================================================
# CONFIG
# ============================================================

SYMBOL = os.getenv("SYMBOL", "ETHUSDT").upper()

# Historia suficiente para contexto y analogías.
# Binance Futures permite hasta 1500 velas por petición; la función pagina.
TF_LIMITS = {
    "5m": int(os.getenv("LIMIT_5M", "800")),
    "15m": int(os.getenv("LIMIT_15M", "3000")),
    "1h": int(os.getenv("LIMIT_1H", "1000")),
    "4h": int(os.getenv("LIMIT_4H", "500")),
}

# Modelo de analogías históricas
ANALOG_K = int(os.getenv("ANALOG_K", "180"))
MIN_ANALOGS = int(os.getenv("MIN_ANALOGS", "80"))

# Filtros de decisión. La probabilidad absoluta se divide entre LONG / SHORT / NEUTRAL,
# por eso no exigimos 60% absoluto a LONG/SHORT. Exigimos ventaja DIRECCIONAL
# condicionada a que el precio salga del escenario neutral.
MIN_DIRECTIONAL_PROB = float(os.getenv("MIN_DIRECTIONAL_PROB", "0.65"))
MAX_NEUTRAL_PROB = float(os.getenv("MAX_NEUTRAL_PROB", "0.45"))
MIN_ABS_EDGE = float(os.getenv("MIN_ABS_EDGE", "0.10"))
MIN_MTF_SCORE = float(os.getenv("MIN_MTF_SCORE", "0.75"))
LOCAL_CONFLICT_SCORE = float(os.getenv("LOCAL_CONFLICT_SCORE", "2.0"))
MIN_ROOM_R = float(os.getenv("MIN_ROOM_R", "1.20"))

# Zona neutral: evita llamar "subida" a movimientos insignificantes
MIN_NEUTRAL_PCT_1H = float(os.getenv("MIN_NEUTRAL_PCT_1H", "0.0015"))  # 0.15%
MIN_NEUTRAL_PCT_6H = float(os.getenv("MIN_NEUTRAL_PCT_6H", "0.0035"))  # 0.35%
NEUTRAL_ATR_FACTOR = float(os.getenv("NEUTRAL_ATR_FACTOR", "0.35"))

# Gestión de entrada / SL / TP
ENTRY_PULLBACK_ATR = float(os.getenv("ENTRY_PULLBACK_ATR", "0.45"))
ENTRY_HALF_WIDTH_ATR = float(os.getenv("ENTRY_HALF_WIDTH_ATR", "0.30"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.00"))
MAX_SL_ATR_MULT = float(os.getenv("MAX_SL_ATR_MULT", "3.50"))
STRUCTURE_BUFFER_ATR = float(os.getenv("STRUCTURE_BUFFER_ATR", "0.35"))

TP1_R_MULT = float(os.getenv("TP1_R_MULT", "1.80"))
TP2_R_MULT = float(os.getenv("TP2_R_MULT", "2.80"))
TP3_R_MULT = float(os.getenv("TP3_R_MULT", "4.00"))

SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "24"))      # 4h en velas de 15m
SR_LOOKBACK = int(os.getenv("SR_LOOKBACK", "96"))            # 24h en velas de 15m
VOL_SPIKE_X = float(os.getenv("VOL_SPIKE_X", "1.50"))
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "20"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

# Ajuste heurístico de la probabilidad histórica por contexto multi-timeframe.
# Es deliberadamente moderado para no "fabricar" probabilidad.
MTF_BONUS_1H = float(os.getenv("MTF_BONUS_1H", "0.08"))
MTF_BONUS_6H = float(os.getenv("MTF_BONUS_6H", "0.06"))

# Fuentes públicas para contratos perpetuos ETHUSDT.
# Orden de preferencia para GitHub Actions:
# 1) OKX ETH-USDT-SWAP
# 2) KuCoin ETHUSDTM
# 3) Bybit ETHUSDT linear perpetual
# 4) Binance USDⓈ-M Futures
OKX_BASE_URL = os.getenv("OKX_BASE_URL", "https://www.okx.com").rstrip("/")
OKX_INST_ID = os.getenv("OKX_INST_ID", "ETH-USDT-SWAP")

KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api-futures.kucoin.com").rstrip("/")
KUCOIN_SYMBOL = os.getenv("KUCOIN_SYMBOL", "ETHUSDTM")

BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

BINANCE_BASE_URLS: List[str] = []
if os.getenv("BINANCE_FUTURES_BASE_URL"):
    BINANCE_BASE_URLS.append(os.getenv("BINANCE_FUTURES_BASE_URL").strip().rstrip("/"))
BINANCE_BASE_URLS += [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]
BINANCE_BASE_URLS = list(dict.fromkeys(BINANCE_BASE_URLS))

OKX_INTERVAL = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
}

KUCOIN_GRANULARITY = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}

BYBIT_INTERVAL = {
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
}

INTERVAL_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}

HTTP_HEADERS = {
    "User-Agent": f"Mozilla/5.0 (ETH_ProbSignal/{VERSION}; +github-actions)",
    "Accept": "application/json",
}

os.environ["SSL_CERT_FILE"] = certifi.where()

# Guarda qué fuente terminó utilizándose en cada timeframe.
DATA_SOURCE_BY_TF: Dict[str, str] = {}


def fetch_btc_perp_price() -> Tuple[Optional[float], Optional[str]]:
    """
    Obtiene el precio actual de BTCUSDT Perpetual con fallbacks públicos.
    No altera el análisis de ETH ni DATA_SOURCE_BY_TF.
    """
    errors: Dict[str, str] = {}

    # 1) OKX BTC-USDT-SWAP
    try:
        response = requests.get(
            f"{OKX_BASE_URL}/api/v5/market/ticker",
            params={"instId": "BTC-USDT-SWAP"},
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if str(payload.get("code")) == "0" and data:
            return float(data[0]["last"]), "OKX BTC-USDT-SWAP"
        raise RuntimeError(f"OKX code={payload.get('code')} msg={payload.get('msg')}")
    except Exception as exc:
        errors["OKX"] = str(exc)
        logger.warning("Fallo precio BTC en OKX: %s", exc)

    # 2) KuCoin Futures (BTC se denomina XBTUSDTM)
    try:
        response = requests.get(
            f"{KUCOIN_BASE_URL}/api/v1/ticker",
            params={"symbol": "XBTUSDTM"},
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        if str(payload.get("code")) == "200000" and data.get("price") is not None:
            return float(data["price"]), "KuCoin XBTUSDTM Perpetual"
        raise RuntimeError(f"KuCoin code={payload.get('code')} msg={payload.get('msg')}")
    except Exception as exc:
        errors["KuCoin"] = str(exc)
        logger.warning("Fallo precio BTC en KuCoin: %s", exc)

    # 3) Bybit USDT Perpetual
    try:
        response = requests.get(
            f"{BYBIT_BASE_URL}/v5/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        data = ((payload.get("result") or {}).get("list") or [])
        if payload.get("retCode") == 0 and data:
            return float(data[0]["lastPrice"]), "Bybit BTCUSDT Perpetual"
        raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")
    except Exception as exc:
        errors["Bybit"] = str(exc)
        logger.warning("Fallo precio BTC en Bybit: %s", exc)

    # 4) Binance USDⓈ-M Futures
    for base in BINANCE_BASE_URLS:
        try:
            response = requests.get(
                f"{base}/fapi/v1/ticker/price",
                params={"symbol": "BTCUSDT"},
                headers=HTTP_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("price") is not None:
                return float(payload["price"]), f"Binance Futures ({base})"
        except Exception as exc:
            errors[f"Binance:{base}"] = str(exc)
            logger.warning("Fallo precio BTC en Binance %s: %s", base, exc)

    detail = " | ".join(f"{name}: {err}" for name, err in errors.items())
    logger.warning("No se pudo obtener precio BTCUSDT Perpetual. %s", detail)
    return None, None


# ============================================================
# TELEGRAM
# ============================================================

def _clean_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


TELEGRAM_TOKEN = _clean_env("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _clean_env("TELEGRAM_CHAT_ID")


# ============================================================
# DATA - PERPETUAL FUTURES (OKX / KUCOIN / BYBIT / BINANCE)
# ============================================================

def _response_preview(response: requests.Response, max_chars: int = 180) -> str:
    try:
        body = response.text.replace("\n", " ").replace("\r", " ").strip()
    except Exception:
        body = ""
    return body[:max_chars]


def _finalize_klines(rows: List[list], interval: str, total_limit: int, source: str) -> pd.DataFrame:
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise RuntimeError(f"{source}: no se recibieron velas {interval}.")

    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base", "taker_buy_quote",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # Excluye la vela aún abierta.
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now_utc].copy()
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    if len(df) < 250:
        raise RuntimeError(
            f"{source}: histórico insuficiente en {interval}: {len(df)} velas cerradas."
        )

    df.set_index("close_time", inplace=True)
    DATA_SOURCE_BY_TF[interval] = source

    logger.info("Datos %s: %d velas cerradas | fuente=%s", interval, len(df), source)
    return df.tail(total_limit).copy()


def _fetch_okx_klines(symbol: str, interval: str, total_limit: int) -> pd.DataFrame:
    """Descarga klines históricos del perpetual ETH-USDT-SWAP de OKX."""
    if interval not in OKX_INTERVAL:
        raise ValueError(f"Intervalo no soportado por OKX: {interval}")

    rows: List[list] = []
    remaining = int(total_limit)
    after: Optional[int] = None
    previous_oldest: Optional[int] = None
    url = f"{OKX_BASE_URL}/api/v5/market/history-candles"

    while remaining > 0:
        page_limit = min(100, remaining)  # OKX history-candles: máximo 100
        params = {
            "instId": OKX_INST_ID,
            "bar": OKX_INTERVAL[interval],
            "limit": str(page_limit),
        }
        if after is not None:
            params["after"] = str(after)

        last_err = None
        payload = None
        for attempt in range(3):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=HTTP_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 429:
                    time.sleep(1 + attempt)
                    continue
                response.raise_for_status()
                try:
                    payload = response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"OKX devolvió respuesta no JSON (HTTP {response.status_code}): "
                        f"{_response_preview(response)}"
                    ) from exc

                if str(payload.get("code")) != "0":
                    raise RuntimeError(
                        f"OKX code={payload.get('code')} msg={payload.get('msg')}"
                    )
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Fallo OKX %s (intento %d/3): %s",
                    interval,
                    attempt + 1,
                    str(exc),
                )
                time.sleep(1 + attempt)
        else:
            raise RuntimeError(f"OKX {interval}: {last_err}")

        page = (payload or {}).get("data") or []
        if not page:
            if rows:
                break
            raise RuntimeError(f"OKX {interval}: respuesta sin velas.")

        normalized: List[list] = []
        for item in page:
            if len(item) < 9:
                continue
            # [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]
            if str(item[8]) != "1":
                continue
            start_ms = int(item[0])
            close_ms = start_ms + INTERVAL_MS[interval] - 1
            normalized.append([
                start_ms,
                item[1], item[2], item[3], item[4], item[5],
                close_ms,
                item[7],
                0, 0.0, 0.0, 0,
            ])

        if normalized:
            rows.extend(normalized)

        oldest_open = min(int(item[0]) for item in page)
        if previous_oldest is not None and oldest_open >= previous_oldest:
            break
        previous_oldest = oldest_open
        after = oldest_open
        remaining = max(0, total_limit - len(rows))

        if len(page) < page_limit:
            break
        # OKX history endpoint: 20 req / 2 s por IP. 0.12s deja margen.
        time.sleep(0.12)

    return _finalize_klines(rows, interval, total_limit, "OKX ETH-USDT-SWAP")


def _fetch_kucoin_klines(symbol: str, interval: str, total_limit: int) -> pd.DataFrame:
    """Descarga klines del perpetual ETHUSDTM de KuCoin Futures."""
    if interval not in KUCOIN_GRANULARITY:
        raise ValueError(f"Intervalo no soportado por KuCoin: {interval}")

    rows: List[list] = []
    remaining = int(total_limit)
    end_time = int(time.time() * 1000)
    previous_oldest: Optional[int] = None
    url = f"{KUCOIN_BASE_URL}/api/v1/kline/query"

    while remaining > 0:
        page_limit = min(500, remaining)
        params = {
            "symbol": KUCOIN_SYMBOL,
            "granularity": KUCOIN_GRANULARITY[interval],
            "to": end_time,
        }

        last_err = None
        payload = None
        for attempt in range(3):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=HTTP_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 429:
                    time.sleep(1 + attempt)
                    continue
                response.raise_for_status()
                try:
                    payload = response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"KuCoin devolvió respuesta no JSON (HTTP {response.status_code}): "
                        f"{_response_preview(response)}"
                    ) from exc

                if str(payload.get("code")) != "200000":
                    raise RuntimeError(
                        f"KuCoin code={payload.get('code')} msg={payload.get('msg')}"
                    )
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Fallo KuCoin %s (intento %d/3): %s",
                    interval,
                    attempt + 1,
                    str(exc),
                )
                time.sleep(1 + attempt)
        else:
            raise RuntimeError(f"KuCoin {interval}: {last_err}")

        page = (payload or {}).get("data") or []
        if not page:
            if rows:
                break
            raise RuntimeError(f"KuCoin {interval}: respuesta sin velas.")

        normalized: List[list] = []
        for item in page:
            if len(item) < 7:
                continue
            # Futures REST clásico: [ts, open, high, low, close, volume, turnover]
            start_ms = int(item[0])
            close_ms = start_ms + INTERVAL_MS[interval] - 1
            normalized.append([
                start_ms,
                item[1], item[2], item[3], item[4], item[5],
                close_ms,
                item[6],
                0, 0.0, 0.0, 0,
            ])

        rows.extend(normalized)
        oldest_open = min(int(item[0]) for item in page)
        if previous_oldest is not None and oldest_open >= previous_oldest:
            break
        previous_oldest = oldest_open
        end_time = oldest_open - 1
        remaining = max(0, total_limit - len(rows))

        if len(page) < page_limit:
            break
        time.sleep(0.12)

    return _finalize_klines(rows, interval, total_limit, "KuCoin ETHUSDTM Perpetual")


def _fetch_bybit_klines(symbol: str, interval: str, total_limit: int) -> pd.DataFrame:
    """Descarga klines del contrato USDT Perpetual de Bybit (category=linear)."""
    if interval not in BYBIT_INTERVAL:
        raise ValueError(f"Intervalo no soportado por Bybit: {interval}")

    rows: List[list] = []
    remaining = int(total_limit)
    end_time: Optional[int] = None
    url = f"{BYBIT_BASE_URL}/v5/market/kline"

    while remaining > 0:
        page_limit = min(1000, remaining)
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": BYBIT_INTERVAL[interval],
            "limit": page_limit,
        }
        if end_time is not None:
            params["end"] = int(end_time)

        last_err = None
        payload = None
        for attempt in range(3):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=HTTP_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"Bybit devolvió respuesta no JSON (HTTP {response.status_code}): "
                        f"{_response_preview(response)}"
                    ) from exc

                if payload.get("retCode") != 0:
                    raise RuntimeError(
                        f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}"
                    )
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Fallo Bybit %s (intento %d/3): %s",
                    interval,
                    attempt + 1,
                    str(exc),
                )
                time.sleep(1 + attempt)
        else:
            raise RuntimeError(f"Bybit {interval}: {last_err}")

        page = (((payload or {}).get("result") or {}).get("list") or [])
        if not page:
            raise RuntimeError(f"Bybit {interval}: respuesta sin velas.")

        # Bybit devuelve [startTime, open, high, low, close, volume, turnover]
        # y ordena de la vela más reciente a la más antigua.
        normalized: List[list] = []
        for item in page:
            start_ms = int(item[0])
            close_ms = start_ms + INTERVAL_MS[interval] - 1
            normalized.append([
                start_ms,
                item[1], item[2], item[3], item[4], item[5],
                close_ms,
                item[6],
                0, 0.0, 0.0, 0,
            ])

        rows.extend(normalized)
        oldest_open = min(int(item[0]) for item in page)
        end_time = oldest_open - 1
        remaining -= len(page)

        if len(page) < page_limit:
            break
        time.sleep(0.08)

    return _finalize_klines(rows, interval, total_limit, "Bybit USDT Perpetual")


def _request_binance_page(
    symbol: str,
    interval: str,
    limit: int,
    end_time: Optional[int] = None,
) -> Tuple[List[list], str]:
    last_err = None
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1500),
    }
    if end_time is not None:
        params["endTime"] = int(end_time)

    for base in BINANCE_BASE_URLS:
        url = f"{base}/fapi/v1/klines"
        for attempt in range(2):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=HTTP_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 451:
                    raise RuntimeError(f"HTTP 451 desde {base}")
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "2"))
                    time.sleep(max(1, retry_after))
                    continue
                response.raise_for_status()
                try:
                    data = response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"respuesta no JSON HTTP {response.status_code}: {_response_preview(response)}"
                    ) from exc
                if not isinstance(data, list) or not data:
                    raise RuntimeError(f"respuesta vacía desde {base}")
                return data, base
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Fallo Binance Futures %s %s (intento %d/2): %s",
                    interval, base, attempt + 1, str(exc),
                )
                time.sleep(1 + attempt)

    raise RuntimeError(f"Binance {interval}: {last_err}")


def _fetch_binance_klines(symbol: str, interval: str, total_limit: int) -> pd.DataFrame:
    rows: List[list] = []
    remaining = int(total_limit)
    end_time = None
    source_base = None

    while remaining > 0:
        page_limit = min(1500, remaining)
        page, source_base = _request_binance_page(
            symbol=symbol,
            interval=interval,
            limit=page_limit,
            end_time=end_time,
        )
        rows.extend(page)
        first_open_time = int(page[0][0])
        end_time = first_open_time - 1
        remaining -= len(page)
        if len(page) < page_limit:
            break
        time.sleep(0.08)

    return _finalize_klines(rows, interval, total_limit, f"Binance Futures ({source_base})")


def fetch_futures_klines(symbol: str, interval: str, total_limit: int) -> pd.DataFrame:
    """
    Intenta cuatro fuentes públicas e independientes, todas de contratos perpetuos:
    OKX -> KuCoin -> Bybit -> Binance.
    """
    errors: Dict[str, str] = {}

    sources = [
        ("OKX", _fetch_okx_klines),
        ("KuCoin", _fetch_kucoin_klines),
        ("Bybit", _fetch_bybit_klines),
        ("Binance", _fetch_binance_klines),
    ]

    for source_name, fetcher in sources:
        try:
            return fetcher(symbol, interval, total_limit)
        except Exception as exc:
            errors[source_name] = str(exc)
            logger.warning("%s no disponible para %s: %s", source_name, interval, exc)

    detail = " | ".join(f"{name}: {err}" for name, err in errors.items())
    raise RuntimeError(
        f"No se pudieron obtener klines PERP {interval} desde ninguna fuente. {detail}"
    )


# ============================================================
# INDICADORES
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()
    df["ema200"] = EMAIndicator(close=close, window=200).ema_indicator()

    macd = MACD(close=close)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["rsi14"] = RSIIndicator(close=close, window=14).rsi()
    df["atr14"] = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=14,
    ).average_true_range()

    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # Features continuos
    safe_close = df["close"].replace(0, np.nan)
    safe_atr = df["atr14"].replace(0, np.nan)
    safe_vol = df["vol_ma20"].replace(0, np.nan)

    df["atr_pct"] = df["atr14"] / safe_close
    df["ema20_gap"] = (df["close"] - df["ema20"]) / safe_close
    df["ema50_gap"] = (df["close"] - df["ema50"]) / safe_close
    df["ema200_gap"] = (df["close"] - df["ema200"]) / safe_close
    df["macd_atr"] = df["macd_diff"] / safe_atr
    df["vol_ratio"] = (df["volume"] / safe_vol).clip(upper=8.0)

    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret6"] = df["close"].pct_change(6)

    df["body_pct"] = (df["close"] - df["open"]) / safe_close
    df["range_pct"] = (df["high"] - df["low"]) / safe_close

    return df.dropna().copy()


# ============================================================
# MULTI-TIMEFRAME
# ============================================================

def timeframe_score(row: pd.Series) -> float:
    """
    Score aproximado -5 a +5.
    Positivo = contexto alcista; negativo = bajista.
    """
    score = 0.0

    score += 1.0 if row["close"] > row["ema20"] else -1.0
    score += 1.0 if row["ema20"] > row["ema50"] else -1.0
    score += 1.0 if row["ema50"] > row["ema200"] else -1.0
    score += 1.0 if row["macd_diff"] > 0 else -1.0

    if row["rsi14"] >= 55:
        score += 1.0
    elif row["rsi14"] <= 45:
        score -= 1.0

    return score


def build_mtf_context(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    scores = {
        tf: timeframe_score(df.iloc[-1])
        for tf, df in data.items()
    }

    # Pesamos más 4h y 1h para no dejar que el ruido de 5m mande la señal.
    mtf_score = (
        0.05 * scores["5m"]
        + 0.20 * scores["15m"]
        + 0.35 * scores["1h"]
        + 0.40 * scores["4h"]
    )

    if mtf_score >= 2.0:
        bias = "ALCISTA FUERTE"
    elif mtf_score >= 0.75:
        bias = "ALCISTA"
    elif mtf_score <= -2.0:
        bias = "BAJISTA FUERTE"
    elif mtf_score <= -0.75:
        bias = "BAJISTA"
    else:
        bias = "NEUTRAL / MIXTO"

    return {
        "score_5m": scores["5m"],
        "score_15m": scores["15m"],
        "score_1h": scores["1h"],
        "score_4h": scores["4h"],
        "mtf_score": mtf_score,
        "bias": bias,
    }


# ============================================================
# ANALOGÍAS HISTÓRICAS
# ============================================================

ANALOG_FEATURES = [
    "rsi14",
    "ema20_gap",
    "ema50_gap",
    "ema200_gap",
    "macd_atr",
    "atr_pct",
    "vol_ratio",
    "ret1",
    "ret3",
    "ret6",
    "body_pct",
    "range_pct",
]

FEATURE_WEIGHTS = np.array(
    [0.8, 1.1, 1.0, 0.8, 1.1, 1.0, 0.7, 0.8, 0.9, 1.0, 0.6, 0.6],
    dtype=float,
)


def _weighted_probs(
    future_returns: np.ndarray,
    distances: np.ndarray,
    neutral_threshold: float,
) -> Tuple[float, float, float, float]:
    """
    Probabilidades ponderadas por similitud.
    """
    median_distance = float(np.median(distances))
    scale = max(median_distance, 1e-6)
    weights = np.exp(-distances / scale)

    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        weights = np.ones_like(distances, dtype=float)
        weight_sum = float(weights.sum())

    up_mask = future_returns > neutral_threshold
    down_mask = future_returns < -neutral_threshold
    neutral_mask = ~(up_mask | down_mask)

    p_up = float(weights[up_mask].sum() / weight_sum)
    p_down = float(weights[down_mask].sum() / weight_sum)
    p_neutral = float(weights[neutral_mask].sum() / weight_sum)
    exp_ret = float(np.average(future_returns, weights=weights))

    return p_up, p_neutral, p_down, exp_ret


def historical_analog_projection(
    df15: pd.DataFrame,
    horizon_bars: int,
    min_neutral_pct: float,
) -> Dict[str, float]:
    """
    Busca estados históricos de 15m parecidos al estado ACTUAL.
    El forward return se calcula solo para las filas históricas.
    No existe el bug de usar el estado de hace 1h/6h como estado actual.
    """
    if len(df15) < 400 + horizon_bars:
        raise RuntimeError("No hay suficiente histórico 15m para analogías.")

    current = df15.iloc[-1]
    current_vector = current[ANALOG_FEATURES].astype(float).to_numpy()

    hist = df15.copy()
    hist["future_return"] = (
        hist["close"].shift(-horizon_bars) / hist["close"] - 1.0
    )

    # Excluimos filas cuyo futuro todavía no existe.
    hist = hist.dropna(subset=["future_return"]).copy()
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(
        subset=ANALOG_FEATURES + ["future_return"]
    )

    if len(hist) < MIN_ANALOGS:
        raise RuntimeError(
            f"Muy pocas observaciones históricas para analogías: {len(hist)}"
        )

    matrix = hist[ANALOG_FEATURES].astype(float).to_numpy()

    # Estandarización robusta suficiente para KNN simple.
    center = np.nanmedian(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    scale[scale < 1e-9] = 1.0

    z_hist = (matrix - center) / scale
    z_current = (current_vector - center) / scale

    diff = (z_hist - z_current) * FEATURE_WEIGHTS
    distances = np.sqrt(np.mean(diff ** 2, axis=1))

    k = min(ANALOG_K, len(hist))
    if k < MIN_ANALOGS:
        raise RuntimeError(
            f"Solo hay {k} analogías, mínimo configurado: {MIN_ANALOGS}"
        )

    nearest_idx = np.argpartition(distances, k - 1)[:k]
    nearest_dist = distances[nearest_idx]
    nearest_returns = hist["future_return"].to_numpy()[nearest_idx]

    # Zona neutral dependiente de la volatilidad actual y del horizonte.
    current_atr_pct = float(current["atr_pct"])
    vol_threshold = (
        NEUTRAL_ATR_FACTOR
        * current_atr_pct
        * math.sqrt(max(horizon_bars, 1))
    )
    neutral_threshold = max(float(min_neutral_pct), float(vol_threshold))

    p_up, p_neutral, p_down, exp_ret = _weighted_probs(
        future_returns=nearest_returns,
        distances=nearest_dist,
        neutral_threshold=neutral_threshold,
    )

    # Cuantiles del retorno de los análogos (no ponderados, deliberadamente simples).
    q25, q50, q75 = np.quantile(nearest_returns, [0.25, 0.50, 0.75])

    return {
        "p_up_raw": p_up,
        "p_neutral_raw": p_neutral,
        "p_down_raw": p_down,
        "expected_return_raw": exp_ret,
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "neutral_threshold": neutral_threshold,
        "sample_size": int(k),
        "median_distance": float(np.median(nearest_dist)),
    }


def adjust_projection_with_mtf(
    projection: Dict[str, float],
    mtf_score: float,
    max_bonus: float,
) -> Dict[str, float]:
    """
    Ajuste moderado de probabilidades según contexto 4h/1h/15m/5m.
    Mantiene la suma en 1.0.
    """
    strength = float(np.clip(mtf_score / 5.0, -1.0, 1.0))
    bonus = strength * max_bonus

    p_up = max(0.0, projection["p_up_raw"] + bonus)
    p_down = max(0.0, projection["p_down_raw"] - bonus)
    p_neutral = max(0.0, projection["p_neutral_raw"])

    total = p_up + p_neutral + p_down
    if total <= 0:
        p_up = p_neutral = p_down = 1.0 / 3.0
    else:
        p_up /= total
        p_neutral /= total
        p_down /= total

    result = dict(projection)
    result.update(
        {
            "p_up": p_up,
            "p_neutral": p_neutral,
            "p_down": p_down,
            "mtf_bonus": bonus,
        }
    )
    return result


# ============================================================
# SOPORTES, RESISTENCIAS Y BREAKOUT
# ============================================================

def support_resistance(df15: pd.DataFrame) -> Tuple[float, float]:
    lookback = min(SR_LOOKBACK, len(df15))
    window = df15.iloc[-lookback:]
    support = float(window["low"].min())
    resistance = float(window["high"].max())
    return support, resistance


def detect_breakout(df15: pd.DataFrame) -> Optional[str]:
    if len(df15) < BREAKOUT_LOOKBACK + 5:
        return None

    last = df15.iloc[-1]
    previous = df15.iloc[-BREAKOUT_LOOKBACK - 1 : -1]

    prior_high = float(previous["high"].max())
    prior_low = float(previous["low"].min())
    prior_vol = float(previous["volume"].mean())

    vol_spike = (
        prior_vol > 0 and float(last["volume"]) >= VOL_SPIKE_X * prior_vol
    )

    if float(last["close"]) > prior_high and vol_spike:
        pct = (float(last["close"]) / prior_high - 1.0) * 100
        return (
            f"RUPTURA ALCISTA 15m: cierre +{pct:.2f}% sobre máximo "
            f"de {BREAKOUT_LOOKBACK} velas con volumen x"
            f"{float(last['volume']) / prior_vol:.2f}"
        )

    if float(last["close"]) < prior_low and vol_spike:
        pct = (float(last["close"]) / prior_low - 1.0) * 100
        return (
            f"RUPTURA BAJISTA 15m: cierre {pct:.2f}% bajo mínimo "
            f"de {BREAKOUT_LOOKBACK} velas con volumen x"
            f"{float(last['volume']) / prior_vol:.2f}"
        )

    return None


# ============================================================
# TIMING 5M
# ============================================================

def timing_5m(df5: pd.DataFrame, side: str) -> str:
    row = df5.iloc[-1]
    price = float(row["close"])
    atr = float(row["atr14"])
    ema20 = float(row["ema20"])
    rsi = float(row["rsi14"])

    if side == "LONG":
        if rsi >= 72 or price > ema20 + 0.90 * atr:
            return "ESPERAR PULLBACK: 5m está extendido."
        if row["macd_diff"] < 0 and price < ema20:
            return "ESPERAR CONFIRMACIÓN: 5m aún no acompaña el LONG."
        return "APTO: 5m acompaña o no está sobreextendido."

    if side == "SHORT":
        if rsi <= 28 or price < ema20 - 0.90 * atr:
            return "ESPERAR REBOTE: 5m está extendido a la baja."
        if row["macd_diff"] > 0 and price > ema20:
            return "ESPERAR CONFIRMACIÓN: 5m aún no acompaña el SHORT."
        return "APTO: 5m acompaña o no está sobreextendido."

    return "SIN TIMING: no existe setup válido."


# ============================================================
# DECISIÓN Y NIVELES
# ============================================================

def choose_trade(
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    mtf: Dict[str, float],
    proj1h: Dict[str, float],
    proj6h: Dict[str, float],
) -> Dict[str, object]:
    """
    Decide LONG / SHORT / WAIT.

    V2.4 separa dos conceptos:
    1) Probabilidad absoluta: LONG / SHORT / NEUTRAL.
    2) Probabilidad direccional: LONG vs SHORT condicionada a que haya movimiento.

    Esto evita exigir un 60% absoluto a LONG/SHORT cuando existe un escenario neutral
    significativo, pero sigue bloqueando operaciones si el mercado está demasiado lateral.
    """
    p_long = 0.65 * proj1h["p_up"] + 0.35 * proj6h["p_up"]
    p_short = 0.65 * proj1h["p_down"] + 0.35 * proj6h["p_down"]
    p_neutral = 0.65 * proj1h["p_neutral"] + 0.35 * proj6h["p_neutral"]

    directional_mass = p_long + p_short
    if directional_mass > 1e-12:
        p_dir_long = p_long / directional_mass
        p_dir_short = p_short / directional_mass
    else:
        p_dir_long = p_dir_short = 0.50

    side = "LONG" if p_dir_long >= p_dir_short else "SHORT"
    directional_best = max(p_dir_long, p_dir_short)
    abs_edge = abs(p_long - p_short)

    price = float(df5.iloc[-1]["close"])
    atr15 = float(df15.iloc[-1]["atr14"])
    ema20_15 = float(df15.iloc[-1]["ema20"])
    support, resistance = support_resistance(df15)

    timing = timing_5m(df5, side)
    reasons: List[str] = []

    # Filtros estadísticos
    if directional_best < MIN_DIRECTIONAL_PROB:
        reasons.append(
            f"Probabilidad direccional insuficiente ({directional_best*100:.1f}% < "
            f"{MIN_DIRECTIONAL_PROB*100:.0f}%)."
        )

    if p_neutral > MAX_NEUTRAL_PROB:
        reasons.append(
            f"Escenario neutral demasiado alto ({p_neutral*100:.1f}% > "
            f"{MAX_NEUTRAL_PROB*100:.0f}%)."
        )

    if abs_edge < MIN_ABS_EDGE:
        reasons.append(
            f"Ventaja absoluta insuficiente ({abs_edge*100:.1f} pp < "
            f"{MIN_ABS_EDGE*100:.0f} pp)."
        )

    # Coherencia multi-timeframe
    if side == "LONG" and mtf["mtf_score"] < MIN_MTF_SCORE:
        reasons.append("El contexto multi-timeframe no confirma suficientemente el LONG.")
    if side == "SHORT" and mtf["mtf_score"] > -MIN_MTF_SCORE:
        reasons.append("El contexto multi-timeframe no confirma suficientemente el SHORT.")

    # El 15m no debe estar fuertemente en contra. Permitimos una corrección moderada (-1/+1),
    # pero no una estructura local claramente opuesta.
    if side == "LONG" and mtf["score_15m"] <= -LOCAL_CONFLICT_SCORE:
        reasons.append("15m está demasiado bajista para ejecutar un LONG todavía.")
    if side == "SHORT" and mtf["score_15m"] >= LOCAL_CONFLICT_SCORE:
        reasons.append("15m está demasiado alcista para ejecutar un SHORT todavía.")

    # Si 1h y 6h son claramente opuestos, no forzamos operación.
    dir1 = "LONG" if proj1h["p_up"] > proj1h["p_down"] else "SHORT"
    dir6 = "LONG" if proj6h["p_up"] > proj6h["p_down"] else "SHORT"
    if dir1 != dir6:
        strong1 = max(proj1h["p_up"], proj1h["p_down"]) >= 0.55
        strong6 = max(proj6h["p_up"], proj6h["p_down"]) >= 0.55
        if strong1 and strong6:
            reasons.append("1h y 6h están en conflicto.")

    # Timing 5m: ahora sí es un filtro de ejecución, no solo una observación.
    if not timing.startswith("APTO"):
        reasons.append(f"Timing 5m no confirma entrada: {timing}")

    result: Dict[str, object] = {
        "side": "WAIT" if reasons else side,
        "p_long": float(p_long),
        "p_short": float(p_short),
        "p_neutral": float(p_neutral),
        "p_dir_long": float(p_dir_long),
        "p_dir_short": float(p_dir_short),
        "directional_best": float(directional_best),
        "directional_mass": float(directional_mass),
        "best_prob": float(max(p_long, p_short)),
        "edge": float(abs_edge),
        "reasons": reasons,
        "price": price,
        "support": support,
        "resistance": resistance,
        "timing": timing,
    }

    if reasons:
        return result

    swing_window = df15.iloc[-min(SWING_LOOKBACK, len(df15)) :]
    swing_low = float(swing_window["low"].min())
    swing_high = float(swing_window["high"].max())

    # Entrada por zona más amplia. Buscamos pullback hacia EMA20 sin perseguir precio.
    if side == "LONG":
        raw_center = max(ema20_15, price - ENTRY_PULLBACK_ATR * atr15)
        center = float(np.clip(raw_center, price - 0.90 * atr15, price + 0.10 * atr15))

        entry_low = center - ENTRY_HALF_WIDTH_ATR * atr15
        entry_high = center + ENTRY_HALF_WIDTH_ATR * atr15

        structural_stop = swing_low - STRUCTURE_BUFFER_ATR * atr15
        structural_distance = center - structural_stop
        max_stop_distance = MAX_SL_ATR_MULT * atr15

        if structural_distance > max_stop_distance:
            result["side"] = "WAIT"
            result["reasons"].append(
                f"SL estructural demasiado amplio ({structural_distance/atr15:.2f} ATR > "
                f"máximo {MAX_SL_ATR_MULT:.2f} ATR)."
            )
            return result

        stop_distance = max(SL_ATR_MULT * atr15, structural_distance)
        room_to_barrier = resistance - center if resistance > center else float("inf")
        room_r = room_to_barrier / stop_distance if math.isfinite(room_to_barrier) else float("inf")

        if room_r < MIN_ROOM_R:
            result["side"] = "WAIT"
            result["reasons"].append(
                f"Resistencia demasiado cerca ({room_r:.2f}R < mínimo {MIN_ROOM_R:.2f}R)."
            )
            result["room_r"] = float(room_r)
            return result

        sl = center - stop_distance
        tp1 = center + TP1_R_MULT * stop_distance
        tp2 = center + TP2_R_MULT * stop_distance
        tp3 = center + TP3_R_MULT * stop_distance

    else:
        raw_center = min(ema20_15, price + ENTRY_PULLBACK_ATR * atr15)
        center = float(np.clip(raw_center, price - 0.10 * atr15, price + 0.90 * atr15))

        entry_low = center - ENTRY_HALF_WIDTH_ATR * atr15
        entry_high = center + ENTRY_HALF_WIDTH_ATR * atr15

        structural_stop = swing_high + STRUCTURE_BUFFER_ATR * atr15
        structural_distance = structural_stop - center
        max_stop_distance = MAX_SL_ATR_MULT * atr15

        if structural_distance > max_stop_distance:
            result["side"] = "WAIT"
            result["reasons"].append(
                f"SL estructural demasiado amplio ({structural_distance/atr15:.2f} ATR > "
                f"máximo {MAX_SL_ATR_MULT:.2f} ATR)."
            )
            return result

        stop_distance = max(SL_ATR_MULT * atr15, structural_distance)
        room_to_barrier = center - support if support < center else float("inf")
        room_r = room_to_barrier / stop_distance if math.isfinite(room_to_barrier) else float("inf")

        if room_r < MIN_ROOM_R:
            result["side"] = "WAIT"
            result["reasons"].append(
                f"Soporte demasiado cerca ({room_r:.2f}R < mínimo {MIN_ROOM_R:.2f}R)."
            )
            result["room_r"] = float(room_r)
            return result

        sl = center + stop_distance
        tp1 = center - TP1_R_MULT * stop_distance
        tp2 = center - TP2_R_MULT * stop_distance
        tp3 = center - TP3_R_MULT * stop_distance

    result.update(
        {
            "entry_center": float(center),
            "entry_low": float(min(entry_low, entry_high)),
            "entry_high": float(max(entry_low, entry_high)),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "risk_usdt": float(stop_distance),
            "risk_pct": float(stop_distance / center) if center else 0.0,
            "rr_tp1": TP1_R_MULT,
            "rr_tp2": TP2_R_MULT,
            "rr_tp3": TP3_R_MULT,
            "room_r": float(room_r),
        }
    )

    return result

# ============================================================
# MENSAJE
# ============================================================

def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_price(x: float) -> str:
    return f"{x:,.2f}"


def projection_direction(p: Dict[str, float]) -> str:
    if p["p_up"] >= p["p_down"] and p["p_up"] >= p["p_neutral"]:
        return "ALCISTA"
    if p["p_down"] >= p["p_up"] and p["p_down"] >= p["p_neutral"]:
        return "BAJISTA"
    return "NEUTRAL"


def compose_message(
    data: Dict[str, pd.DataFrame],
    mtf: Dict[str, float],
    proj1h: Dict[str, float],
    proj6h: Dict[str, float],
    trade: Dict[str, object],
    breakout: Optional[str],
    btc_price: Optional[float],
) -> str:
    tz = pytz.timezone("America/Bogota")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    last5 = data["5m"].iloc[-1]
    last15 = data["15m"].iloc[-1]
    price = float(last5["close"])

    # Banda central (Q25-Q75) de los análogos históricos, expresada en precio.
    range1_low = price * (1.0 + float(proj1h["q25"]))
    range1_high = price * (1.0 + float(proj1h["q75"]))
    range6_low = price * (1.0 + float(proj6h["q25"]))
    range6_high = price * (1.0 + float(proj6h["q75"]))

    lines: List[str] = []

    lines.append(f"📊 ETHUSDT PERP — ProbSignal {VERSION}")
    lines.append(f"Fecha/Hora: {now}")
    sources = sorted(set(DATA_SOURCE_BY_TF.values()))
    source_text = " / ".join(sources) if sources else "Perpetual Futures"
    lines.append(f"Fuente: {source_text} | solo velas cerradas")
    lines.append(f"Precio: {fmt_price(price)}")
    if btc_price is not None:
        lines.append(f"BTCUSDT PERP: {fmt_price(btc_price)}")
    else:
        lines.append("BTCUSDT PERP: no disponible")

    if breakout:
        lines.append(f"🚨 {breakout}")

    lines.append("")
    lines.append("🧭 CONTEXTO MULTI-TIMEFRAME")
    lines.append(
        f"4H {mtf['score_4h']:+.1f} | "
        f"1H {mtf['score_1h']:+.1f} | "
        f"15m {mtf['score_15m']:+.1f} | "
        f"5m {mtf['score_5m']:+.1f}"
    )
    lines.append(f"Sesgo: {mtf['bias']} (score combinado {mtf['mtf_score']:+.2f})")

    lines.append("")
    lines.append("⏰ PROYECCIÓN 1 HORA")
    lines.append(
        f"{projection_direction(proj1h)} | "
        f"Alza {pct(proj1h['p_up'])} | "
        f"Neutral {pct(proj1h['p_neutral'])} | "
        f"Baja {pct(proj1h['p_down'])}"
    )
    lines.append(
        f"Retorno esperado: {proj1h['expected_return_raw']*100:+.2f}% | "
        f"Mediana: {proj1h['q50']*100:+.2f}% | n={proj1h['sample_size']}"
    )
    lines.append(
        f"Rango central 50%: {fmt_price(min(range1_low, range1_high))} – "
        f"{fmt_price(max(range1_low, range1_high))}"
    )

    lines.append("")
    lines.append("⏰ PROYECCIÓN 6 HORAS")
    lines.append(
        f"{projection_direction(proj6h)} | "
        f"Alza {pct(proj6h['p_up'])} | "
        f"Neutral {pct(proj6h['p_neutral'])} | "
        f"Baja {pct(proj6h['p_down'])}"
    )
    lines.append(
        f"Retorno esperado: {proj6h['expected_return_raw']*100:+.2f}% | "
        f"Mediana: {proj6h['q50']*100:+.2f}% | n={proj6h['sample_size']}"
    )
    lines.append(
        f"Rango central 50%: {fmt_price(min(range6_low, range6_high))} – "
        f"{fmt_price(max(range6_low, range6_high))}"
    )

    lines.append("")
    lines.append(
        f"ATR15m: {fmt_price(float(last15['atr14']))} "
        f"({float(last15['atr_pct'])*100:.2f}%)"
    )
    lines.append(
        f"Soporte 24h aprox.: {fmt_price(float(trade['support']))} | "
        f"Resistencia 24h aprox.: {fmt_price(float(trade['resistance']))}"
    )

    lines.append("")
    lines.append("🎯 PROBABILIDAD COMBINADA")
    lines.append(
        f"Absoluta: LONG {pct(float(trade['p_long']))} | "
        f"SHORT {pct(float(trade['p_short']))} | "
        f"Neutral {pct(float(trade['p_neutral']))}"
    )
    lines.append(
        f"Si sale del rango: LONG {pct(float(trade['p_dir_long']))} | "
        f"SHORT {pct(float(trade['p_dir_short']))}"
    )

    lines.append("")
    if trade["side"] == "WAIT":
        lines.append("⚪ NO TRADE / ESPERAR")
        for reason in trade["reasons"]:
            lines.append(f"• {reason}")
        lines.append(f"Timing 5m: {trade['timing']}")
        lines.append("No se ejecuta hasta que probabilidad, estructura y timing coincidan.")

    else:
        emoji = "🟢" if trade["side"] == "LONG" else "🔴"
        lines.append(f"{emoji} SETUP {trade['side']}")
        lines.append(
            f"Confianza direccional: {pct(float(trade['directional_best']))} | "
            f"Neutral: {pct(float(trade['p_neutral']))} | "
            f"edge abs.: {float(trade['edge'])*100:.1f} pp"
        )
        lines.append(
            f"Zona entrada: {fmt_price(float(trade['entry_low']))} – "
            f"{fmt_price(float(trade['entry_high']))}"
        )
        lines.append(f"Centro referencia: {fmt_price(float(trade['entry_center']))}")
        lines.append(
            f"SL: {fmt_price(float(trade['sl']))} | "
            f"riesgo {fmt_price(float(trade['risk_usdt']))} USDT "
            f"({float(trade['risk_pct'])*100:.2f}%)"
        )
        lines.append(f"TP1: {fmt_price(float(trade['tp1']))}  (R {trade['rr_tp1']:.1f})")
        lines.append(f"TP2: {fmt_price(float(trade['tp2']))}  (R {trade['rr_tp2']:.1f})")
        lines.append(f"TP3: {fmt_price(float(trade['tp3']))}  (R {trade['rr_tp3']:.1f})")
        if math.isfinite(float(trade.get("room_r", float("inf")))):
            lines.append(f"Espacio hasta barrera 24h: {float(trade['room_r']):.2f}R")
        lines.append(f"Timing 5m: {trade['timing']}")

    return "\n".join(lines)

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text: str) -> Tuple[bool, Optional[str]]:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "TELEGRAM_TOKEN o TELEGRAM_CHAT_ID no configurados."

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.ok:
            return True, None
        return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as exc:
        return False, str(exc)


# ============================================================
# PIPELINE
# ============================================================

def analyze_once() -> str:
    data: Dict[str, pd.DataFrame] = {}

    for tf, limit in TF_LIMITS.items():
        raw = fetch_futures_klines(
            symbol=SYMBOL,
            interval=tf,
            total_limit=limit,
        )
        data[tf] = add_indicators(raw)

    mtf = build_mtf_context(data)

    # 15m -> 1h = 4 velas; 6h = 24 velas
    proj1h_raw = historical_analog_projection(
        data["15m"],
        horizon_bars=4,
        min_neutral_pct=MIN_NEUTRAL_PCT_1H,
    )
    proj6h_raw = historical_analog_projection(
        data["15m"],
        horizon_bars=24,
        min_neutral_pct=MIN_NEUTRAL_PCT_6H,
    )

    proj1h = adjust_projection_with_mtf(
        proj1h_raw,
        mtf_score=mtf["mtf_score"],
        max_bonus=MTF_BONUS_1H,
    )
    proj6h = adjust_projection_with_mtf(
        proj6h_raw,
        mtf_score=mtf["mtf_score"],
        max_bonus=MTF_BONUS_6H,
    )

    trade = choose_trade(
        df5=data["5m"],
        df15=data["15m"],
        mtf=mtf,
        proj1h=proj1h,
        proj6h=proj6h,
    )

    breakout = detect_breakout(data["15m"])
    btc_price, _btc_source = fetch_btc_perp_price()

    return compose_message(
        data=data,
        mtf=mtf,
        proj1h=proj1h,
        proj6h=proj6h,
        trade=trade,
        breakout=breakout,
        btc_price=btc_price,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger.info("Iniciando ETH ProbSignal %s", VERSION)

    try:
        message = analyze_once()
        print(message)

        sent, err = send_telegram(message)
        if sent:
            logger.info("✅ Señal %s enviada a Telegram.", VERSION)
        else:
            logger.warning("Telegram no enviado: %s", err)

    except Exception as exc:
        logger.exception("Error en ETH ProbSignal V2")

        tz = pytz.timezone("America/Bogota")
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        heartbeat = (
            f"🔔 Heartbeat ETH ProbSignal {VERSION}\n"
            f"Fecha/Hora: {now}\n"
            f"Motivo: {exc}"
        )
        send_telegram(heartbeat)
        raise SystemExit(1)