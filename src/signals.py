import numpy as np

def generate_signals(df, fib_levels=None):
    signals = []
    confidences = []
    watch_signals = []

    # Get Fib levels for easy access
    fib_618 = fib_levels['61.8%'] if fib_levels else None
    fib_382 = fib_levels['38.2%'] if fib_levels else None

    for i in range(1, len(df)):
        # Enhanced BB+MACD+OBV logic
        obv_ma = df['obv'].rolling(14).mean().iloc[i]
        buy_conditions = [
            df['Close'].iloc[i] < df['bb_lower'].iloc[i],
            df['macd_diff'].iloc[i] > 0,
            df['obv'].iloc[i] > obv_ma
        ]
        # Secondary filters (not primary triggers)
        if df['rsi'].iloc[i] < 35:
            buy_conditions.append(True)
        if df['Close'].iloc[i] > df['ichimoku_a'].iloc[i] and df['Close'].iloc[i] > df['ichimoku_b'].iloc[i]:
            buy_conditions.append(True)

        sell_conditions = [
            df['Close'].iloc[i] > df['bb_upper'].iloc[i],
            df['macd_diff'].iloc[i] < 0,
            df['obv'].iloc[i] < obv_ma
        ]
        if df['rsi'].iloc[i] > 65:
            sell_conditions.append(True)
        if df['Close'].iloc[i] < df['ichimoku_a'].iloc[i] and df['Close'].iloc[i] < df['ichimoku_b'].iloc[i]:
            sell_conditions.append(True)

        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)

        # Watchlist logic (secondary layer)
        watch_buy = False
        watch_sell = False
        if fib_618 and fib_382:
            # Near Lower BB + Near Fib 61.8%
            watch_buy = (
                (df['Close'].iloc[i] < df['bb_lower'].iloc[i] * 1.01) and
                (abs(df['Close'].iloc[i] - fib_618) / fib_618 < 0.01)
            )
            # Near Upper BB + Near Fib 38.2%
            watch_sell = (
                (df['Close'].iloc[i] > df['bb_upper'].iloc[i] * 0.99) and
                (abs(df['Close'].iloc[i] - fib_382) / fib_382 < 0.01)
            )

        # Assign signals
        if buy_score >= 3 and buy_score > sell_score:
            signals.append('Buy')
        elif sell_score >= 3 and sell_score > buy_score:
            signals.append('Sell')
        elif buy_score >= 3 and sell_score >= 3 and buy_score == sell_score:
            signals.append('Ambiguous')
        else:
            signals.append('Hold')

        # Watchlist signals
        if watch_buy:
            watch_signals.append('watch_buy')
        elif watch_sell:
            watch_signals.append('watch_sell')
        else:
            watch_signals.append('')

        # Confidence score as a percentage
        max_conditions = 5  # 3 primary + 2 secondary for both buy and sell
        confidence = int(100 * (buy_score if buy_score > sell_score else sell_score) / max_conditions)
        confidences.append(confidence)

    # Pad first row
    df['signal'] = ['Hold'] + signals
    df['watch_signal'] = [''] + watch_signals
    df['confidence'] = [0] + confidences

    # Final signal: strict if present, else watchlist
    df['final_signal'] = np.where(df['signal'] != 'Hold', df['signal'], df['watch_signal'])
    return df

