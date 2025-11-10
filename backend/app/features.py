from collections import deque
from typing import Deque, Optional, Dict
import math


class FeatureCalculator:
    """Maintains rolling state for feature calculations.

    Methods accept the latest close/high/low/open and produce a dict of
    precomputed features to store with the candle.
    """

    def __init__(self):
        self.closes: Deque[float] = deque(maxlen=400)
        self.opens: Deque[float] = deque(maxlen=400)
        self.highs: Deque[float] = deque(maxlen=400)
        self.lows: Deque[float] = deque(maxlen=400)
        self.volumes: Deque[float] = deque(maxlen=400)
        self.returns: Deque[float] = deque(maxlen=400)
        self.tr: Deque[float] = deque(maxlen=400)  # true range values
        self.prev_close: Optional[float] = None
        self.ema_12: Optional[float] = None
        self.ema_26: Optional[float] = None
        self.macd_signal: Optional[float] = None
        self.rsi_avg_gain: Optional[float] = None
        self.rsi_avg_loss: Optional[float] = None
        # State for OBV and MFI flows
        self.obv: float = 0.0
        self.pos_flow: Deque[float] = deque(maxlen=100)
        self.neg_flow: Deque[float] = deque(maxlen=100)
        # For multi-period RSI and CCI
        self.rsi7_gain: Optional[float] = None
        self.rsi7_loss: Optional[float] = None
        self.rsi21_gain: Optional[float] = None
        self.rsi21_loss: Optional[float] = None
        # For VWAP
        self.tp_vol_20: Deque[float] = deque(maxlen=50)
        self.vol_agg_20: Deque[float] = deque(maxlen=50)
        # Close z-score tracking
        self.close_window_20: Deque[float] = deque(maxlen=20)
        # Historical closes for ROC / MACD already in closes

    def update(self, open_: float, high: float, low: float, close: float, volume: float = 0.0) -> Dict[str, float]:
        features: Dict[str, float] = {}

        # Basic sequences
        self.opens.append(open_)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)

        # Returns
        if self.prev_close is not None and self.prev_close != 0:
            ret = (close - self.prev_close) / self.prev_close
            self.returns.append(ret)
        else:
            ret = 0.0
        self.prev_close = close

        # Rolling returns windows
        def rolling_ret(n: int) -> Optional[float]:
            if len(self.closes) >= n:
                first = self.closes[-n]
                last = self.closes[-1]
                if first != 0:
                    return (last - first) / first
            return None

        features["ret_1"] = ret
        features["ret_5"] = rolling_ret(5) or 0.0
        features["ret_15"] = rolling_ret(15) or 0.0

        # MA
        def sma(n: int) -> Optional[float]:
            if len(self.closes) >= n:
                return sum(list(self.closes)[-n:]) / n
            return None

        features["ma_5"] = sma(5) or 0.0
        features["ma_20"] = sma(20) or 0.0
        features["ma_50"] = sma(50) or 0.0
        features["ma_200"] = sma(200) or 0.0

        # EMA
        def ema(current: float, prev: Optional[float], period: int) -> float:
            k = 2 / (period + 1)
            return current * k + (prev if prev is not None else current) * (1 - k)

        self.ema_12 = ema(close, self.ema_12, 12)
        self.ema_26 = ema(close, self.ema_26, 26)
        features["ema_12"] = self.ema_12
        features["ema_26"] = self.ema_26
        # Additional EMAs
        self.ema_9 = ema(close, getattr(self, 'ema_9', None), 9)
        self.ema_50 = ema(close, getattr(self, 'ema_50', None), 50)
        features["ema_9"] = self.ema_9
        features["ema_50"] = self.ema_50

        # MACD (12,26,9)
        if self.ema_12 is not None and self.ema_26 is not None:
            macd = self.ema_12 - self.ema_26
            self.macd_signal = ema(macd, self.macd_signal, 9)
            features["macd"] = macd
            features["macd_signal"] = self.macd_signal if self.macd_signal is not None else macd
            features["macd_hist"] = macd - (self.macd_signal if self.macd_signal is not None else macd)
        else:
            features["macd"], features["macd_signal"], features["macd_hist"] = 0.0, 0.0, 0.0

        # RSI (14)
        if len(self.closes) >= 2:
            change = self.closes[-1] - self.closes[-2]
            gain = max(change, 0)
            loss = abs(min(change, 0))
            period = 14
            if self.rsi_avg_gain is None or self.rsi_avg_loss is None:
                # Initialize using simple average of first period if enough data
                if len(self.closes) >= period + 1:
                    gains = [max(self.closes[i+1] - self.closes[i], 0) for i in range(-period-1, -1)]
                    losses = [abs(min(self.closes[i+1] - self.closes[i], 0)) for i in range(-period-1, -1)]
                    self.rsi_avg_gain = sum(gains) / period
                    self.rsi_avg_loss = sum(losses) / period
            else:
                self.rsi_avg_gain = (self.rsi_avg_gain * (period - 1) + gain) / period
                self.rsi_avg_loss = (self.rsi_avg_loss * (period - 1) + loss) / period

            if self.rsi_avg_gain is not None and self.rsi_avg_loss is not None and self.rsi_avg_loss != 0:
                rs = self.rsi_avg_gain / self.rsi_avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
        else:
            rsi = 50.0
        features["rsi_14"] = rsi
        # RSI 7 & 21 using same logic with Wilder smoothing
        for period, g_attr, l_attr, out_name in [
            (7, 'rsi7_gain', 'rsi7_loss', 'rsi_7'),
            (21, 'rsi21_gain', 'rsi21_loss', 'rsi_21'),
        ]:
            if len(self.closes) >= 2:
                change = self.closes[-1] - self.closes[-2]
                gain = max(change, 0)
                loss = abs(min(change, 0))
                avg_gain = getattr(self, g_attr)
                avg_loss = getattr(self, l_attr)
                if avg_gain is None or avg_loss is None:
                    if len(self.closes) >= period + 1:
                        gains = [max(self.closes[i+1]-self.closes[i], 0) for i in range(-period-1, -1)]
                        losses = [abs(min(self.closes[i+1]-self.closes[i], 0)) for i in range(-period-1, -1)]
                        avg_gain = sum(gains)/period
                        avg_loss = sum(losses)/period
                else:
                    avg_gain = (avg_gain * (period - 1) + gain) / period
                    avg_loss = (avg_loss * (period - 1) + loss) / period
                setattr(self, g_attr, avg_gain)
                setattr(self, l_attr, avg_loss)
                if avg_loss and avg_loss != 0:
                    rsx = avg_gain / avg_loss
                    rsi_p = 100 - (100 / (1 + rsx))
                else:
                    rsi_p = 50.0
            else:
                rsi_p = 50.0
            features[out_name] = rsi_p

        # True Range & ATR(14)
        if len(self.closes) >= 2:
            prev_close = self.closes[-2]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            self.tr.append(tr)
        if len(self.tr) >= 14:
            features["atr_14"] = sum(list(self.tr)[-14:]) / 14
        else:
            features["atr_14"] = 0.0

        # Volatility (std of returns last 20)
        if len(self.returns) >= 20:
            r20 = list(self.returns)[-20:]
            mean_r = sum(r20) / 20
            var = sum((x - mean_r) ** 2 for x in r20) / 20
            features["vol_20"] = math.sqrt(var)
        else:
            features["vol_20"] = 0.0
        # vol_50
        if len(self.returns) >= 50:
            r50 = list(self.returns)[-50:]
            mean_r50 = sum(r50) / 50
            var50 = sum((x - mean_r50) ** 2 for x in r50) / 50
            features["vol_50"] = math.sqrt(var50)
        else:
            features["vol_50"] = 0.0

        # Candle anatomy
        body = abs(close - open_)
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        features["body"] = body
        features["upper_shadow"] = upper_shadow
        features["lower_shadow"] = lower_shadow

        # Bollinger Bands (20,2)
        if len(self.closes) >= 20:
            last20 = list(self.closes)[-20:]
            mean20 = sum(last20) / 20
            var20 = sum((x - mean20) ** 2 for x in last20) / 20
            std20 = math.sqrt(var20)
            bb_upper = mean20 + 2 * std20
            bb_lower = mean20 - 2 * std20
            features["bb_upper_20_2"] = bb_upper
            features["bb_lower_20_2"] = bb_lower
            if bb_upper != bb_lower:
                features["bb_pct_b_20_2"] = (close - bb_lower) / (bb_upper - bb_lower)
                features["bb_bandwidth_20_2"] = (bb_upper - bb_lower) / mean20 if mean20 != 0 else 0.0
            else:
                features["bb_pct_b_20_2"] = 0.5
                features["bb_bandwidth_20_2"] = 0.0
        else:
            features["bb_upper_20_2"] = 0.0
            features["bb_lower_20_2"] = 0.0
            features["bb_pct_b_20_2"] = 0.5
            features["bb_bandwidth_20_2"] = 0.0
        # Bollinger 50,2 (only pct_b and bandwidth to limit columns)
        if len(self.closes) >= 50:
            last50 = list(self.closes)[-50:]
            mean50 = sum(last50)/50
            var50 = sum((x - mean50)**2 for x in last50)/50
            std50 = math.sqrt(var50)
            upper50 = mean50 + 2*std50
            lower50 = mean50 - 2*std50
            if upper50 != lower50:
                features["bb_pct_b_50_2"] = (close - lower50)/(upper50 - lower50)
                features["bb_bandwidth_50_2"] = (upper50 - lower50)/mean50 if mean50 != 0 else 0.0
            else:
                features["bb_pct_b_50_2"] = 0.5
                features["bb_bandwidth_50_2"] = 0.0
        else:
            features["bb_pct_b_50_2"] = 0.5
            features["bb_bandwidth_50_2"] = 0.0

        # Stochastic %K(14) and %D(3)
        if len(self.closes) >= 14:
            last_high = max(list(self.highs)[-14:])
            last_low = min(list(self.lows)[-14:])
            if last_high != last_low:
                k = (close - last_low) / (last_high - last_low) * 100
            else:
                k = 50.0
            # Smooth %K with 3-period SMA to get %D
            # Keep last 3 k values in returns deque as proxy by appending?
            # We maintain a small local list instead for simplicity
            # Here we approximate %D using last three closes-based %K if available
            k_list = []
            # reconstruct up to three past %K using history (approximation)
            hist = list(self.closes)
            highs = list(self.highs)
            lows = list(self.lows)
            for back in range(0, 3):
                if len(hist) >= 14 + back:
                    segment = hist[-(14 + back): -back or None]
                    hseg = highs[-(14 + back): -back or None]
                    lseg = lows[-(14 + back): -back or None]
                    hh = max(hseg)
                    ll = min(lseg)
                    c = segment[-1]
                    if hh != ll:
                        k_list.append((c - ll) / (hh - ll) * 100)
            d = sum(k_list) / len(k_list) if k_list else k
            features["stoch_k_14_3"] = k
            features["stoch_d_14_3"] = d
        else:
            features["stoch_k_14_3"] = 50.0
            features["stoch_d_14_3"] = 50.0

        # Williams %R(14)
        if len(self.closes) >= 14:
            hh = max(list(self.highs)[-14:])
            ll = min(list(self.lows)[-14:])
            if hh != ll:
                willr = -100 * (hh - close) / (hh - ll)
            else:
                willr = -50.0
        else:
            willr = -50.0
        features["williams_r_14"] = willr

        # OBV
        if len(self.closes) >= 2:
            sign = 1 if self.closes[-1] > self.closes[-2] else (-1 if self.closes[-1] < self.closes[-2] else 0)
            self.obv += sign * volume
        features["obv"] = self.obv

        # MFI(14) & CMF(20)
        if len(self.closes) >= 2:
            typical_price = (high + low + close) / 3.0
            prev_typical_price = (self.highs[-2] + self.lows[-2] + self.closes[-2]) / 3.0 if len(self.closes) >= 2 else typical_price
            pos = typical_price * volume if typical_price > prev_typical_price else 0.0
            neg = typical_price * volume if typical_price < prev_typical_price else 0.0
        else:
            pos = neg = 0.0
        self.pos_flow.append(pos)
        self.neg_flow.append(neg)
        if len(self.pos_flow) >= 14:
            pos_sum = sum(list(self.pos_flow)[-14:])
            neg_sum = sum(list(self.neg_flow)[-14:])
            if neg_sum == 0:
                mfi = 100.0
            else:
                ratio = pos_sum / neg_sum if neg_sum != 0 else 0.0
                mfi = 100 - (100 / (1 + ratio))
        else:
            mfi = 50.0
        features["mfi_14"] = mfi
        # We approximate rolling sums via lists; for precision, a dedicated deque per flow could be maintained
        # CMF(20): (sum(ADL) over 20) / (sum(volume) over 20)
        if len(self.closes) >= 20:
            # Accumulation/Distribution (ADL) factor for last bar
            mfm = ((close - low) - (high - close)) / (high - low) if (high - low) != 0 else 0.0
            adl = mfm * volume
            # Approximate CMF using last 20 elements lists
            vols = list(self.volumes)[-20:]
            highs20 = list(self.highs)[-20:]
            lows20 = list(self.lows)[-20:]
            closes20 = list(self.closes)[-20:]
            adl_sum = 0.0
            for hi, lo, cl, vol in zip(highs20, lows20, closes20, vols):
                mfm_i = ((cl - lo) - (hi - cl)) / (hi - lo) if (hi - lo) != 0 else 0.0
                adl_sum += mfm_i * vol
            vol_sum = sum(vols)
            features["cmf_20"] = adl_sum / vol_sum if vol_sum != 0 else 0.0
        else:
            features["cmf_20"] = 0.0

        # Volume z-score(20)
        if len(self.volumes) >= 20:
            v20 = list(self.volumes)[-20:]
            mean_v = sum(v20) / 20
            var_v = sum((x - mean_v) ** 2 for x in v20) / 20
            std_v = math.sqrt(var_v)
            features["vol_z_20"] = (volume - mean_v) / std_v if std_v != 0 else 0.0
        else:
            features["vol_z_20"] = 0.0

        # Distance to rolling minima / drawdown from rolling max
        for n in (20, 50):
            if len(self.closes) >= n:
                window = list(self.closes)[-n:]
                min_c = min(window)
                features[f"dist_min_close_{n}"] = (close - min_c) / min_c if min_c != 0 else 0.0
            else:
                features[f"dist_min_close_{n}"] = 0.0
        if len(self.closes) >= 20:
            max20 = max(list(self.closes)[-20:])
            features["drawdown_from_max_20"] = (close - max20) / max20 if max20 != 0 else 0.0
        else:
            features["drawdown_from_max_20"] = 0.0
        # dist_min_close_100
        if len(self.closes) >= 100:
            min100 = min(list(self.closes)[-100:])
            features["dist_min_close_100"] = (close - min100)/min100 if min100 != 0 else 0.0
        else:
            features["dist_min_close_100"] = 0.0

        # DI+/DI-/ADX(14) - simplified Wilder's smoothing
        if len(self.closes) >= 2:
            up_move = high - self.highs[-2]
            down_move = self.lows[-2] - low
            plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
            # Append TR for ADX calculation already computed above as last of self.tr
            tr_val = self.tr[-1] if len(self.tr) > 0 else (high - low)
            period = 14
            # Maintain rolling sums approximately
            # For simplicity, compute DI from last 14 bars using lists
            if len(self.highs) >= period + 1:
                highs14 = list(self.highs)[-period-1:]
                lows14 = list(self.lows)[-period-1:]
                closes14 = list(self.closes)[-period-1:]
                plus_dm_sum = 0.0
                minus_dm_sum = 0.0
                tr_sum = 0.0
                for i in range(1, len(highs14)):
                    up = highs14[i] - highs14[i-1]
                    down = lows14[i-1] - lows14[i]
                    plus_dm_sum += up if up > down and up > 0 else 0.0
                    minus_dm_sum += down if down > up and down > 0 else 0.0
                    tr_i = max(highs14[i] - lows14[i], abs(highs14[i] - closes14[i-1]), abs(lows14[i] - closes14[i-1]))
                    tr_sum += tr_i
                plus_di = 100 * (plus_dm_sum / tr_sum) if tr_sum != 0 else 0.0
                minus_di = 100 * (minus_dm_sum / tr_sum) if tr_sum != 0 else 0.0
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0.0
                # Approximate ADX as average of last 14 dx values (recompute from history)
                # For simplicity we set ADX to this dx when insufficient history
                adx = dx
                features["di_plus_14"] = plus_di
                features["di_minus_14"] = minus_di
                features["adx_14"] = adx
            else:
                features["di_plus_14"] = 0.0
                features["di_minus_14"] = 0.0
                features["adx_14"] = 0.0
        else:
            features["di_plus_14"] = 0.0
            features["di_minus_14"] = 0.0
            features["adx_14"] = 0.0

        # Run-length encoding for consecutive up/down closes
        run_up = 0
        run_down = 0
        if len(self.closes) >= 2:
            # count consecutive increases/decreases ending at last
            for i in range(len(self.closes) - 1, 0, -1):
                if self.closes[i] > self.closes[i - 1]:
                    if run_down == 0:
                        run_up += 1
                    else:
                        break
                elif self.closes[i] < self.closes[i - 1]:
                    if run_up == 0:
                        run_down += 1
                    else:
                        break
                else:
                    break
        features["run_up"] = run_up
        features["run_down"] = run_down

        # Normalized body vs range
        range_hl = (high - low)
        features["range_hl"] = range_hl
        features["body_pct_of_range"] = body / range_hl if range_hl != 0 else 0.0

        # ATR 7 (reuse tr deque)
        if len(self.tr) >= 7:
            features["atr_7"] = sum(list(self.tr)[-7:]) / 7
        else:
            features["atr_7"] = 0.0

        # ROC 10
        if len(self.closes) >= 11 and self.closes[-11] != 0:
            features["roc_10"] = (self.closes[-1] - self.closes[-11]) / self.closes[-11]
        else:
            features["roc_10"] = 0.0

        # CCI 20 (Typical Price based)
        if len(self.closes) >= 20:
            tp_list = [ (self.highs[i] + self.lows[i] + self.closes[i]) / 3 for i in range(-20, 0) ]
            tp_mean = sum(tp_list)/20
            mean_dev = sum(abs(tp - tp_mean) for tp in tp_list)/20
            current_tp = (high + low + close)/3
            if mean_dev != 0:
                features["cci_20"] = (current_tp - tp_mean) / (0.015 * mean_dev)
            else:
                features["cci_20"] = 0.0
        else:
            features["cci_20"] = 0.0

        # Price relative to MA 20 & 50
        ma20 = features.get("ma_20", 0.0)
        features["price_to_ma_20"] = (close/ma20 - 1) if ma20 else 0.0
        ma50 = features.get("ma_50", 0.0)
        features["price_to_ma_50"] = (close/ma50 - 1) if ma50 else 0.0

        # VWAP deviations (20 & 50 window) approximate intrabar by closes
        tp = (high + low + close)/3
        self.tp_vol_20.append(tp * volume)
        self.vol_agg_20.append(volume)
        if len(self.tp_vol_20) >= 20:
            vwap20 = sum(list(self.tp_vol_20)[-20:]) / (sum(list(self.vol_agg_20)[-20:]) or 1)
            features["vwap_20_dev"] = (close / vwap20 - 1) if vwap20 else 0.0
        else:
            features["vwap_20_dev"] = 0.0
        if len(self.tp_vol_20) >= 50:
            vwap50 = sum(list(self.tp_vol_20)[-50:]) / (sum(list(self.vol_agg_20)[-50:]) or 1)
            features["vwap_50_dev"] = (close / vwap50 - 1) if vwap50 else 0.0
        else:
            features["vwap_50_dev"] = 0.0

        # Close z-score 20
        self.close_window_20.append(close)
        if len(self.close_window_20) >= 20:
            cw = list(self.close_window_20)
            mean_cw = sum(cw)/len(cw)
            var_cw = sum((x-mean_cw)**2 for x in cw)/len(cw)
            std_cw = math.sqrt(var_cw)
            features["zscore_close_20"] = (close - mean_cw)/std_cw if std_cw else 0.0
        else:
            features["zscore_close_20"] = 0.0

        return features

    # Snapshot & restore -----------------------------------------------
    def snapshot(self) -> Dict[str, any]:
        """Return serializable dict of internal rolling state.
        Only lightweight numerics and list copies; deques converted to lists.
        """
        return {
            "closes": list(self.closes),
            "opens": list(self.opens),
            "highs": list(self.highs),
            "lows": list(self.lows),
            "volumes": list(self.volumes),
            "returns": list(self.returns),
            "tr": list(self.tr),
            "prev_close": self.prev_close,
            "ema_12": self.ema_12,
            "ema_26": self.ema_26,
            "macd_signal": self.macd_signal,
            "rsi_avg_gain": self.rsi_avg_gain,
            "rsi_avg_loss": self.rsi_avg_loss,
            "obv": self.obv,
            "pos_flow": list(self.pos_flow),
            "neg_flow": list(self.neg_flow),
            "rsi7_gain": self.rsi7_gain,
            "rsi7_loss": self.rsi7_loss,
            "rsi21_gain": self.rsi21_gain,
            "rsi21_loss": self.rsi21_loss,
            "tp_vol_20": list(self.tp_vol_20),
            "vol_agg_20": list(self.vol_agg_20),
            "close_window_20": list(self.close_window_20),
            "ema_9": getattr(self, 'ema_9', None),
            "ema_50": getattr(self, 'ema_50', None),
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, any]) -> "FeatureCalculator":
        fc = cls()
        # Assign deques by extending (preserve maxlen)
        for name in ["closes","opens","highs","lows","volumes","returns","tr","pos_flow","neg_flow","tp_vol_20","vol_agg_20","close_window_20"]:
            seq = data.get(name, [])
            dq = getattr(fc, name)
            dq.extend(seq[-dq.maxlen:])
        # Scalars
        for attr in ["prev_close","ema_12","ema_26","macd_signal","rsi_avg_gain","rsi_avg_loss","obv","rsi7_gain","rsi7_loss","rsi21_gain","rsi21_loss","ema_9","ema_50"]:
            if attr in data:
                setattr(fc, attr, data[attr])
        return fc
