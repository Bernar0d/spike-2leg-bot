import os
import time
import requests
import numpy as np
import pandas as pd
from typing import Optional, Dict
from dotenv import load_dotenv
import ccxt

# Load .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EXCHANGE_NAME = os.getenv("EXCHANGE")
SYMBOLS = os.getenv("SYMBOLS").split(",")
TIMEFRAME = os.getenv("TIMEFRAME")
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT"))

# Init exchange
exchange_class = getattr(ccxt, EXCHANGE_NAME)
exchange = exchange_class({"enableRateLimit": True})


# ---------------------------------------------------------
# 📌 Send Telegram message
# ---------------------------------------------------------
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    requests.post(url, json=payload)


# ---------------------------------------------------------
# 📌 Spike 2 Leg Strategy Detection (Advanced Algorithm)
# ---------------------------------------------------------
def detect_s2l(df, 
               lookback=20,
               spike_min_candles=3,
               spike_body_factor=1.8,   # spike body must be X times average body
               spike_range_factor=1.5,  # spike range must be larger than prev ranges * factor
               spike_volume_factor=1.5,  # spike volume must be at least X times average volume of prev 20 candles
               exhaustion_wick_ratio=0.4, # wick relative to body to count as exhaustion
               leg1_min_pct=40, leg1_max_pct=70, # retracement % of spike
               leg2_retest_tol_pct=5.0,  # how close leg2 must get to spike extreme (%)
               leg2_momentum_factor=0.6, # leg2 body avg must be <= this * spike body avg
               prz_wick_ratio=0.3,
               prz_absorption_bars=2,
               prz_volume_divergence=True,  # require volume to decrease at PRZ (volume divergence)
               verbose=False
              ):
    """
    Detects Spike 2-Leg (S2L) in OHLCV dataframe and returns JSON-like dict
    matching required format in the prompt.
    """

    # minimal length
    N = max(lookback + 10, 50)
    if len(df) < N:
        return {
            "setup": "invalid",
            "direction": "none",
            "probability": 0,
            "comment": "Not enough data"
        }

    # consider working window: last lookback+30 candles to capture spike+legs
    recent = df.iloc[-(lookback+50):].copy()
    closes = recent["close"].values
    opens = recent["open"].values
    highs = recent["high"].values
    lows = recent["low"].values
    volumes = recent["volume"].values
    times = recent.index.to_pydatetime()

    # compute body sizes and ranges
    bodies = np.abs(closes - opens)
    body_pct = bodies / opens * 100.0
    ranges = highs - lows

    # find candidate spike: look for the most recent run of >= spike_min_candles same-direction large bodies
    # We'll scan from the end backwards to find last run
    dir_arr = np.sign(closes - opens)  # +1 bullish, -1 bearish, 0 neutral
    # convert zeros to previous direction if any to avoid tie issues
    for i in range(1, len(dir_arr)):
        if dir_arr[i] == 0:
            dir_arr[i] = dir_arr[i-1]

    def analyze_run(end_idx):
        # end_idx inclusive index in recent arrays pointing to last candle of spike
        # find run length of same direction ending at end_idx
        d = dir_arr[end_idx]
        if d == 0:
            return None
        i = end_idx
        run_idx = []
        while i >= 0 and dir_arr[i] == d:
            run_idx.append(i)
            i -= 1
        run_idx = run_idx[::-1]  # chronological
        return {"direction": "bull" if d>0 else "bear", "idxs": run_idx}

    # scan last 40 bars for a candidate end of spike
    candidate = None
    for idx in range(len(recent)-1, max(len(recent)-40, 0), -1):
        run = analyze_run(idx)
        if run and len(run["idxs"]) >= spike_min_candles:
            # compute avg body of prev lookback candles excluding the run
            before_start = run["idxs"][0] - 1
            if before_start - lookback + 1 < 0:
                continue
            prev_bodies = bodies[before_start-lookback+1:before_start+1]
            avg_prev_body = np.mean(prev_bodies) if len(prev_bodies)>0 else 0.0
            run_bodies = bodies[run["idxs"]]
            avg_run_body = np.mean(run_bodies)
            # condition: run bodies significantly larger than previous average bodies
            if avg_prev_body > 0 and avg_run_body >= spike_body_factor * avg_prev_body:
                # range condition: spike range must be clearly larger than previous ranges
                prev_ranges = ranges[before_start-lookback+1:before_start+1]
                if len(prev_ranges)==0:
                    continue
                max_prev_range = np.max(prev_ranges)
                spike_range = np.max(ranges[run["idxs"]])
                if spike_range >= spike_range_factor * max_prev_range:
                    # volume condition: spike volume must be significantly higher than previous average volume
                    prev_volumes = volumes[before_start-lookback+1:before_start+1]
                    if len(prev_volumes) > 0:
                        avg_prev_volume = np.mean(prev_volumes)
                        spike_volumes = volumes[run["idxs"]]
                        avg_spike_volume = np.mean(spike_volumes)
                        if avg_prev_volume > 0 and avg_spike_volume >= spike_volume_factor * avg_prev_volume:
                            candidate = run
                            spike_start_idx = run["idxs"][0]
                            spike_end_idx = run["idxs"][-1]
                            break
                    else:
                        # if no previous volume data, accept candidate anyway
                        candidate = run
                        spike_start_idx = run["idxs"][0]
                        spike_end_idx = run["idxs"][-1]
                        break

    if candidate is None:
        return {"setup": "invalid", "direction": "none", "probability": 0, "comment": "No clear spike (3+ large same-direction candles) found."}

    direction = candidate["direction"]  # 'bull' or 'bear'
    # Extract spike metrics
    spike_bodies = bodies[spike_start_idx:spike_end_idx+1]
    avg_spike_body = float(np.mean(spike_bodies))
    spike_volumes = volumes[spike_start_idx:spike_end_idx+1]
    avg_spike_volume = float(np.mean(spike_volumes))
    spike_high = float(np.max(highs[spike_start_idx:spike_end_idx+1]))
    spike_low = float(np.min(lows[spike_start_idx:spike_end_idx+1]))
    spike_extreme = spike_high if direction == "bull" else spike_low
    spike_range_total = spike_high - spike_low

    # Exhaustion: look at last candle of spike for wick in direction of exhaustion
    last_spike_idx = spike_end_idx
    last_open = opens[last_spike_idx]
    last_close = closes[last_spike_idx]
    last_high = highs[last_spike_idx]
    last_low = lows[last_spike_idx]
    last_body = abs(last_close - last_open)
    if last_body == 0:
        last_body = 1e-9
    if direction == "bull":
        upper_wick = last_high - max(last_close, last_open)
        exhaustion = (upper_wick >= exhaustion_wick_ratio * last_body) or (upper_wick > 0 and (last_close < last_high - last_body*0.2))
    else:
        lower_wick = min(last_close, last_open) - last_low
        exhaustion = (lower_wick >= exhaustion_wick_ratio * last_body) or (lower_wick > 0 and (last_close > last_low + last_body*0.2))

    if not exhaustion:
        return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "Spike found but no visible exhaustion on spike extreme."}

    # Leg1: after spike_end_idx, find a clear correction (clean swing) that retraces 40-70% of spike
    # define spike_anchor price (start of spike) as open of first spike candle
    spike_anchor_price = opens[spike_start_idx]
    if direction == "bull":
        spike_move = spike_extreme - spike_anchor_price
        if spike_move <= 0:
            return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "Invalid spike geometry."}
        # search for leg1 low after spike_end_idx within next ~lookback bars
        search_start = spike_end_idx + 1
        search_end = min(search_start + lookback, len(recent)-1)
        if search_start >= len(recent):
            return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "No bars after spike to form Leg1."}
        leg1_low = float(np.min(lows[search_start:search_end+1]))
        retracement_pct = ( (spike_extreme - leg1_low) / spike_move ) * 100.0
    else:
        spike_move = spike_anchor_price - spike_extreme
        if spike_move <= 0:
            return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "Invalid spike geometry."}
        search_start = spike_end_idx + 1
        search_end = min(search_start + lookback, len(recent)-1)
        if search_start >= len(recent):
            return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "No bars after spike to form Leg1."}
        leg1_high = float(np.max(highs[search_start:search_end+1]))
        retracement_pct = ( (leg1_high - spike_extreme) / spike_move ) * 100.0

    # check Leg1 percent range
    if retracement_pct < leg1_min_pct*0.9 or retracement_pct > leg1_max_pct*1.1:
        return {"setup": "invalid", "direction": direction, "probability": 0, "comment": f"Leg1 retracement not in {leg1_min_pct}-{leg1_max_pct}% (got {retracement_pct:.1f}%)."}

    # ensure Leg1 is a clean swing: use simple criterion - leg1 move is monotonic for at least 2-3 candles (not noise)
    # count monotonic candles from spike_end_idx+1 in direction opposite to spike
    mono_required = 2
    mono_count = 0
    for i in range(search_start, search_end+1):
        if direction == "bull":
            if closes[i] < opens[i]:
                mono_count += 1
            else:
                break
        else:
            if closes[i] > opens[i]:
                mono_count += 1
            else:
                break
    if mono_count < mono_required:
        return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "Leg1 not a clean counter swing (too noisy)."}

    # identify Leg1 end index
    if direction == "bull":
        leg1_end_idx = int(np.argmin(lows[search_start:search_end+1]) + search_start)
        leg1_val = float(lows[leg1_end_idx])
    else:
        leg1_end_idx = int(np.argmax(highs[search_start:search_end+1]) + search_start)
        leg1_val = float(highs[leg1_end_idx])

    # Leg2: after leg1_end_idx we must see a retest towards spike extreme:
    leg2_search_start = leg1_end_idx + 1
    leg2_search_end = min(leg2_search_start + lookback, len(recent)-1)
    if leg2_search_start >= len(recent):
        return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "Leg2 not formed yet (no bars after Leg1)."}

    # find the highest (for bull) or lowest (for bear) price in leg2 window
    if direction == "bull":
        leg2_high = float(np.max(highs[leg2_search_start:leg2_search_end+1]))
        dist_to_spike_pct = ( (spike_extreme - leg2_high) / spike_move ) * 100.0  # how far below spike the retest reached (0% = exact)
        reached_pct = 100.0 - dist_to_spike_pct  # how close to spike in percent (100% = equals spike)
        leg2_idxs = np.where(highs[leg2_search_start:leg2_search_end+1] >= (spike_extreme * (1 - leg2_retest_tol_pct/100.0)))[0]
        leg2_formed = len(leg2_idxs) > 0
    else:
        leg2_low = float(np.min(lows[leg2_search_start:leg2_search_end+1]))
        dist_to_spike_pct = ( (leg2_low - spike_extreme) / spike_move ) * 100.0
        reached_pct = 100.0 - dist_to_spike_pct
        leg2_idxs = np.where(lows[leg2_search_start:leg2_search_end+1] <= (spike_extreme * (1 + leg2_retest_tol_pct/100.0)))[0]
        leg2_formed = len(leg2_idxs) > 0

    if not leg2_formed:
        return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "Leg2 retest toward spike extreme not completed."}

    # check Leg2 momentum weaker than spike: compare average body in leg2 vs avg_spike_body
    # take up to 5 candles around first retest index
    first_retest_offset = int(leg2_idxs[0])
    first_retest_idx = leg2_search_start + first_retest_offset
    window_start = max(leg2_search_start, first_retest_idx - 3)
    window_end = min(len(recent)-1, first_retest_idx + 3)
    leg2_bodies = bodies[window_start:window_end+1]
    avg_leg2_body = float(np.mean(leg2_bodies)) if len(leg2_bodies)>0 else 0.0

    momentum_weaker = (avg_leg2_body <= leg2_momentum_factor * avg_spike_body)

    if not momentum_weaker:
        return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "Leg2 momentum not weaker than spike (must be weaker)."}

    # PRZ confirmation: at retest area we need at least one of:
    # - rejection wick (in opposite direction) on a retest candle
    # - absorption (several opposite bars absorbing)
    # - Break of minor structure (BOS) — we approximate BOS as a failure to make new high (for bull) and a subsequent break of local swing
    # - Volume divergence: volume at retest should be lower than spike volume
    prz_confirmed = False
    prz_reason = ""
    volume_divergence_ok = False

    # check volume divergence at PRZ (retest volume should be lower than spike volume)
    if prz_volume_divergence:
        retest_volumes = volumes[max(leg2_search_start, first_retest_idx - 2):first_retest_idx + 1]
        if len(retest_volumes) > 0:
            avg_retest_volume = float(np.mean(retest_volumes))
            # volume divergence: retest volume should be significantly lower than spike volume
            if avg_spike_volume > 0 and avg_retest_volume < avg_spike_volume * 0.8:
                volume_divergence_ok = True

    # check wick rejection at first_retest_idx
    idx = first_retest_idx
    o,c,h,l = opens[idx], closes[idx], highs[idx], lows[idx]
    body = abs(c-o) if abs(c-o)>0 else 1e-9
    if direction == "bull":
        upper_wick = h - max(c,o)
        if upper_wick >= prz_wick_ratio * body:
            prz_confirmed = True
            prz_reason = "rejection_wick"
    else:
        lower_wick = min(c,o) - l
        if lower_wick >= prz_wick_ratio * body:
            prz_confirmed = True
            prz_reason = "rejection_wick"

    # check absorption: sequence of prz_absorption_bars candles pushing into spike extreme but with decreasing bodies
    if not prz_confirmed:
        seq_start = max(leg2_search_start, first_retest_idx - prz_absorption_bars + 1)
        seq_bodies = bodies[seq_start:first_retest_idx+1]
        if len(seq_bodies) >= prz_absorption_bars:
            # absorption if bodies decreasing and final candle shows rejection (small body + wick)
            if np.all(np.diff(seq_bodies) < 0):
                prz_confirmed = True
                prz_reason = "absorption"

    # simple BOS approximation: after retest there's an opposite-direction candle that breaks a recent minor swing (local high/low)
    if not prz_confirmed:
        # get a small local swing before retest (3 bars)
        pre_swing_start = max(0, first_retest_idx - 6)
        pre_swing_high = float(np.max(highs[pre_swing_start:first_retest_idx])) if pre_swing_start < first_retest_idx else None
        pre_swing_low = float(np.min(lows[pre_swing_start:first_retest_idx])) if pre_swing_start < first_retest_idx else None
        # check after retest if price breaks pre_swing in opposite direction (i.e., bearish BOS after bull retest)
        post_idx_end = min(len(recent)-1, first_retest_idx + 6)
        if direction == "bull" and pre_swing_high is not None:
            # if after retest there is a candle that closes below a small structure low -> BOS for reversal
            post_lows = lows[first_retest_idx:post_idx_end+1]
            if len(post_lows)>0 and np.min(post_lows) < (leg1_val if direction=="bull" else leg1_val):
                prz_confirmed = True
                prz_reason = "BOS"
        elif direction == "bear" and pre_swing_low is not None:
            post_highs = highs[first_retest_idx:post_idx_end+1]
            if len(post_highs)>0 and np.max(post_highs) > (leg1_val if direction=="bear" else leg1_val):
                prz_confirmed = True
                prz_reason = "BOS"

    # Volume divergence can confirm PRZ even if other signals are weak
    if not prz_confirmed and volume_divergence_ok:
        prz_confirmed = True
        prz_reason = "volume_divergence"

    if not prz_confirmed:
        return {"setup": "invalid", "direction": direction, "probability": 0, "comment": "No PRZ confirmation (rejection/absorption/BOS/volume_divergence) found at retest."}

    # --- Probability scoring according to the given weights ---
    # Spike quality (0-30): based on avg_run_body relative to prev avg body and range dominance and exhaustion presence and volume
    # compute prev avg body for normalization
    prev_idx_end = spike_start_idx - 1
    prev_idx_start = max(0, prev_idx_end - lookback + 1)
    prev_avg_body = np.mean(bodies[prev_idx_start:prev_idx_end+1]) if prev_idx_end - prev_idx_start + 1 > 0 else 1e-9
    prev_avg_volume = np.mean(volumes[prev_idx_start:prev_idx_end+1]) if prev_idx_end - prev_idx_start + 1 > 0 else 1e-9
    spike_score = 0.0
    # body ratio score
    body_ratio = avg_spike_body / (prev_avg_body + 1e-9)
    body_score = min(1.0, (body_ratio - spike_body_factor + 1.0))  # rough scaling
    body_score = max(0.0, body_score)
    # range dominance
    prev_max_range = float(np.max(ranges[prev_idx_start:prev_idx_end+1])) if prev_idx_end - prev_idx_start + 1 > 0 else 1e-9
    range_ratio = spike_range_total / (prev_max_range + 1e-9)
    range_score = min(1.0, (range_ratio - spike_range_factor + 1.0))
    range_score = max(0.0, range_score)
    # volume score: how much spike volume exceeds previous average
    volume_ratio = avg_spike_volume / (prev_avg_volume + 1e-9) if prev_avg_volume > 0 else 1.0
    volume_score = min(1.0, (volume_ratio - spike_volume_factor + 1.0) / 2.0)  # normalize to 0-1
    volume_score = max(0.0, volume_score)
    # exhaustion bonus
    exhaustion_score = 1.0 if exhaustion else 0.0
    spike_score = (0.5 * body_score + 0.25 * range_score + 0.15 * volume_score + 0.1 * exhaustion_score) * 30.0  # scale 0-30

    # Leg1 clarity (0-30): closeness of retracement to 40-70 and clean swing length
    ideal_mid = (leg1_min_pct + leg1_max_pct) / 2.0
    # gaussian-ish score based on distance from ideal range
    dist = 0.0
    if retracement_pct < leg1_min_pct:
        dist = leg1_min_pct - retracement_pct
    elif retracement_pct > leg1_max_pct:
        dist = retracement_pct - leg1_max_pct
    else:
        dist = abs(retracement_pct - ideal_mid)
    # map dist to 0-1 (smaller better)
    leg1_score = max(0.0, 1.0 - (dist / ( (leg1_max_pct - leg1_min_pct)/2.0 + 1e-9 )))
    # also reward monotonicity (mono_count)
    mono_score = min(1.0, mono_count / 4.0)
    leg1_score = (0.7 * leg1_score + 0.3 * mono_score) * 30.0

    # Leg2 clarity (0-20): existence and momentum weaker
    # reward how close the retest reached (% reached_pct), and momentum weaker
    reached_normalized = min(1.0, reached_pct / 100.0)
    momentum_score = 1.0 if momentum_weaker else 0.0
    leg2_score = (0.7 * reached_normalized + 0.3 * momentum_score) * 20.0

    # PRZ quality (0-20): rejection wick, absorption, BOS, volume divergence
    prz_score = 0.0
    if prz_reason == "rejection_wick":
        prz_score = 18.0
        if volume_divergence_ok:
            prz_score += 2.0  # bonus for volume divergence
    elif prz_reason == "absorption":
        prz_score = 14.0
        if volume_divergence_ok:
            prz_score += 4.0  # bonus for volume divergence
    elif prz_reason == "BOS":
        prz_score = 16.0
        if volume_divergence_ok:
            prz_score += 3.0  # bonus for volume divergence
    elif prz_reason == "volume_divergence":
        prz_score = 12.0  # volume divergence alone is less strong but still valid
    else:
        prz_score = 0.0
    
    # cap at 20
    prz_score = min(20.0, prz_score)

    total = spike_score + leg1_score + leg2_score + prz_score
    total = max(0.0, min(100.0, total))

    volume_info = f", vol_div={'✓' if volume_divergence_ok else '✗'}" if prz_volume_divergence else ""
    comment = f"Spike detected ({direction}), retrace {retracement_pct:.1f}%, leg2 retest reached ≈{reached_pct:.1f}%, PRZ={prz_reason}{volume_info}."

    result = {
        "setup": "valid" if total >= 50.0 else "invalid",  # you can tune threshold for 'valid'
        "direction": "long" if direction=="bull" else "short",
        "probability": round(total, 2),
        "comment": comment
    }
    return result


# ---------------------------------------------------------
# 📌 Main Scanner Loop
# ---------------------------------------------------------
def run_bot():
    send_telegram("🚀 Spike 2Leg Bot Started.")

    while True:
        for symbol in SYMBOLS:
            try:
                # Fetch OHLCV data from exchange
                candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_LIMIT)
                
                # Convert to DataFrame (CCXT format: [timestamp, open, high, low, close, volume])
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Use advanced detection algorithm
                result = detect_s2l(df)

                if result["setup"] == "valid" and result["probability"] >= 60:
                    msg = f"""
📌 *Spike-2Leg Signal Detected*
Symbol: {symbol}
Direction: {result['direction']}
Probability: {result['probability']}%
Comment: {result['comment']}
                    """
                    send_telegram(msg)

            except Exception as e:
                send_telegram(f"⚠️ Error on {symbol}: {str(e)}")

        time.sleep(10)  # Scan every 10 seconds


if __name__ == "__main__":
    run_bot()
