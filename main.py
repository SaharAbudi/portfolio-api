from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# -----------------------------
# ⚙️ הגדרות בסיס
# -----------------------------

TICKERS = [
    "SPY",       # S&P500 ETF
    "QQQM",      # Nasdaq 100 ETF
    "IWM",       # Russell 2000 ETF
    "^GSPC",     # S&P 500 Index
    "^IXIC",     # Nasdaq Composite Index
    "^VIX",      # Volatility Index
    "GC=F",      # Gold Futures
]

FRONTIER_ASSETS = ["SPY", "QQQM", "IWM", "GC=F"]
RISK_FREE_RATE = 0.03  # 3% לשנה

# תיקים מוגדרים מראש
PORTFOLIOS = {
    "Balanced (50% SPY / 30% QQQM / 20% Gold)": {
        "SPY": 0.50,
        "QQQM": 0.30,
        "GC=F": 0.20
    },
    "Tech Growth (70% QQQM / 20% SPY / 10% Gold)": {
        "QQQM": 0.70,
        "SPY": 0.20,
        "GC=F": 0.10
    },
    "Defensive (60% SPY / 30% Gold / 10% QQQM)": {
        "SPY": 0.60,
        "GC=F": 0.30,
        "QQQM": 0.10
    },
    "Conservative (70% SPY / 20% Gold / 10% QQQM)": {
        "SPY": 0.70,
        "GC=F": 0.20,
        "QQQM": 0.10
    },
    "Growth-Value Mix (40% SPY / 40% QQQM / 20% Gold)": {
        "SPY": 0.40,
        "QQQM": 0.40,
        "GC=F": 0.20
    },
    "Ultra Tech (85% QQQM / 10% SPY / 5% Gold)": {
        "QQQM": 0.85,
        "SPY": 0.10,
        "GC=F": 0.05
    },
    "Risk Protect (50% SPY / 40% Gold / 10% QQQM)": {
        "SPY": 0.50,
        "GC=F": 0.40,
        "QQQM": 0.10
    }
}


# -----------------------------
# 🧮 פונקציות עזר פיננסיות
# -----------------------------

def portfolio_metrics(daily_series: pd.Series, risk_free: float = RISK_FREE_RATE):
    """חישוב מדדים לתיק על בסיס סדרת תשואה יומית."""
    annual_return = daily_series.mean() * 252
    volatility = daily_series.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free) / volatility if volatility != 0 else np.nan

    cumulative = (1 + daily_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    ratio = annual_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Return/DD Ratio": ratio
    }


def build_portfolio_from_weights(weights, assets, daily_returns):
    """בונה סדרת רווח יומית מתיק עם משקולות."""
    w = pd.Series(weights, index=assets)
    return daily_returns[assets].dot(w)


# -----------------------------
# 🧠 Cache Manager
# -----------------------------

class DataCache:
    def __init__(self):
        # נכסים
        self.prices_df: pd.DataFrame | None = None
        self.daily_returns: pd.DataFrame | None = None
        self.cumulative: pd.DataFrame | None = None
        self.assets_summary: pd.DataFrame | None = None

        # תיקים
        self.portfolio_daily: dict[str, pd.Series] = {}
        self.portfolio_cumulative: dict[str, pd.Series] = {}
        self.portfolio_summary: pd.DataFrame | None = None
        self.best_portfolios: dict | None = None

        # Efficient frontier
        self.frontier_points: dict | None = None
        self.optimal_summary: pd.DataFrame | None = None
        self.optimal_curves: dict[str, pd.Series] = {}

        # זמנים
        self.last_assets_update: datetime | None = None
        self.last_frontier_update: datetime | None = None

    # ----------- נכסים ותיקים -----------

    def ensure_core_data(self, force: bool = False, assets_ttl_hours: int = 6):
        """טוען/מרענן נתוני נכסים ותיקים."""
        now = datetime.utcnow()
        if (
            force
            or self.prices_df is None
            or self.last_assets_update is None
            or (now - self.last_assets_update) > timedelta(hours=assets_ttl_hours)
        ):
            self._load_core_data()
            self.last_assets_update = now

    def _load_core_data(self):
        # 1. הורדת נתונים מ-yfinance
        dfs = []
        for t in TICKERS:
            temp = yf.download(t, period="5y", auto_adjust=True, progress=False)
            if temp is None or temp.empty:
                continue
            close_series = temp["Close"]
            close_series.name = t
            dfs.append(close_series)

        if not dfs:
            raise RuntimeError("לא הצלחנו להוריד נתונים מ-yfinance")

        prices_df = pd.concat(dfs, axis=1, join="inner").sort_index()
        daily_returns = prices_df.pct_change().dropna()
        cumulative = (1 + daily_returns).cumprod()

        # מדדי נכסים
        annual_return = daily_returns.mean() * 252
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = (annual_return - RISK_FREE_RATE) / volatility
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        assets_summary = pd.DataFrame({
            "Annual Return (%)": annual_return * 100,
            "Volatility (%)": volatility * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_drawdown * 100
        }).round(3)

        # שמירה ב-cache
        self.prices_df = prices_df
        self.daily_returns = daily_returns
        self.cumulative = cumulative
        self.assets_summary = assets_summary

        # 2. חישוב תיקים
        self._calc_portfolios()

    def _calc_portfolios(self):
        if self.daily_returns is None:
            raise RuntimeError("daily_returns לא מוגדרת")

        daily = self.daily_returns
        portfolio_results = {}
        portfolio_daily = {}
        portfolio_cumulative = {}

        for name, weights in PORTFOLIOS.items():
            w = []
            for col in daily.columns:
                w.append(weights.get(col, 0))
            w = np.array(w)

            port_daily = daily.dot(w)
            port_cumulative = (1 + port_daily).cumprod()

            portfolio_daily[name] = port_daily
            portfolio_cumulative[name] = port_cumulative

            portfolio_results[name] = portfolio_metrics(port_daily)

        portfolio_df = pd.DataFrame(portfolio_results).T

        # המרה לאחוזים
        portfolio_df["Annual Return"] *= 100
        portfolio_df["Volatility"] *= 100
        portfolio_df["Max Drawdown"] *= 100
        portfolio_df = portfolio_df.round(3)

        # דירוגים
        portfolio_df["Return Rank"] = portfolio_df["Annual Return"].rank(ascending=False)
        portfolio_df["Risk Rank"] = portfolio_df["Volatility"].rank(ascending=True)
        portfolio_df["Sharpe Rank"] = portfolio_df["Sharpe"].rank(ascending=False)
        portfolio_df["Drawdown Rank"] = portfolio_df["Max Drawdown"].rank(ascending=True)
        portfolio_df["R/DD Rank"] = portfolio_df["Return/DD Ratio"].rank(ascending=False)

        portfolio_df = portfolio_df.astype({
            "Return Rank": int,
            "Risk Rank": int,
            "Sharpe Rank": int,
            "Drawdown Rank": int,
            "R/DD Rank": int
        })

        # תיקים הכי טובים
        best_sharpe = portfolio_df["Sharpe"].idxmax()
        best_return = portfolio_df["Annual Return"].idxmax()
        best_rdd = portfolio_df["Return/DD Ratio"].idxmax()
        lowest_vol = portfolio_df["Volatility"].idxmin()

        self.portfolio_daily = portfolio_daily
        self.portfolio_cumulative = portfolio_cumulative
        self.portfolio_summary = portfolio_df
        self.best_portfolios = {
            "best_sharpe": best_sharpe,
            "best_return": best_return,
            "best_return_dd_ratio": best_rdd,
            "lowest_volatility": lowest_vol,
        }

    # ----------- Efficient Frontier -----------

    def ensure_frontier(self, force: bool = False, frontier_ttl_hours: int = 24):
        now = datetime.utcnow()
        if (
            force
            or self.frontier_points is None
            or self.last_frontier_update is None
            or (now - self.last_frontier_update) > timedelta(hours=frontier_ttl_hours)
        ):
            self._calc_frontier()
            self.last_frontier_update = now

    def _calc_frontier(self, num_portfolios: int = 10000):
        if self.daily_returns is None:
            raise RuntimeError("daily_returns לא מוגדרת")

        frontier_returns = self.daily_returns[FRONTIER_ASSETS]
        asset_expected_returns = frontier_returns.mean() * 252
        asset_cov = frontier_returns.cov() * 252

        portfolio_weights = []
        portfolio_returns = []
        portfolio_vols = []
        portfolio_sharpes = []

        for _ in range(num_portfolios):
            w = np.random.random(len(FRONTIER_ASSETS))
            w = w / np.sum(w)

            portfolio_weights.append(w)

            ret = np.dot(w, asset_expected_returns)
            portfolio_returns.append(ret)

            vol = np.sqrt(np.dot(w.T, np.dot(asset_cov, w)))
            portfolio_vols.append(vol)

            sharpe = (ret - RISK_FREE_RATE) / vol if vol != 0 else np.nan
            portfolio_sharpes.append(sharpe)

        portfolio_returns = np.array(portfolio_returns)
        portfolio_vols = np.array(portfolio_vols)
        portfolio_sharpes = np.array(portfolio_sharpes)

        idx_max_sharpe = int(np.nanargmax(portfolio_sharpes))
        idx_min_vol = int(np.nanargmin(portfolio_vols))

        max_sharpe_weights = portfolio_weights[idx_max_sharpe]
        min_vol_weights = portfolio_weights[idx_min_vol]

        # טבלאות משקולות
        max_sharpe_df = pd.DataFrame({
            "Asset": FRONTIER_ASSETS,
            "Weight": max_sharpe_weights
        })

        min_vol_df = pd.DataFrame({
            "Asset": FRONTIER_ASSETS,
            "Weight": min_vol_weights
        })

        # מדדי ביצוע
        max_sharpe_daily = build_portfolio_from_weights(
            max_sharpe_weights, FRONTIER_ASSETS, self.daily_returns
        )
        min_vol_daily = build_portfolio_from_weights(
            min_vol_weights, FRONTIER_ASSETS, self.daily_returns
        )

        opt_summary = {
            "Efficient Max Sharpe": portfolio_metrics(max_sharpe_daily),
            "Efficient Min Volatility": portfolio_metrics(min_vol_daily)
        }
        opt_df = pd.DataFrame(opt_summary).T
        opt_df["Annual Return"] *= 100
        opt_df["Volatility"] *= 100
        opt_df["Max Drawdown"] *= 100
        opt_df = opt_df.round(3)

        # שמירת curves
        max_sharpe_curve = (1 + max_sharpe_daily).cumprod()
        min_vol_curve = (1 + min_vol_daily).cumprod()

        self.frontier_points = {
            "returns": portfolio_returns.tolist(),
            "volatility": portfolio_vols.tolist(),
            "sharpe": portfolio_sharpes.tolist(),
        }
        self.optimal_summary = opt_df
        self.optimal_curves = {
            "Efficient Max Sharpe": max_sharpe_curve,
            "Efficient Min Volatility": min_vol_curve,
        }
        self.optimal_weights = {
            "Efficient Max Sharpe": max_sharpe_df.to_dict(orient="records"),
            "Efficient Min Volatility": min_vol_df.to_dict(orient="records"),
        }


# -----------------------------
# 🚀 FastAPI App
# -----------------------------

app = FastAPI(
    title="Portfolio Analytics API",
    version="1.0.0",
    description="API אנליטי לתיקים ונכסים – מבוסס על yfinance וניתוח 5 שנים אחורה."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # במידת הצורך תצמצם לדומיין שלך
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = DataCache()


# -----------------------------
# 🧪 Endpoints
# -----------------------------

@app.on_event("startup")
def startup_event():
    """טוען נתונים ראשוניים בזמן עליית השרת."""
    try:
        cache.ensure_core_data()
        cache.ensure_frontier()
    except Exception as e:
        # לא מפיל את השרת, רק רושם לוג (ב-Render תראה בלוגים)
        print(f"[startup] Failed to load initial data: {e}")


@app.get("/status")
def status():
    """מצב המערכת + זמנים אחרונים של cache."""
    return {
        "ok": True,
        "tickers": TICKERS,
        "frontier_assets": FRONTIER_ASSETS,
        "risk_free_rate": RISK_FREE_RATE,
        "last_assets_update": cache.last_assets_update.isoformat() if cache.last_assets_update else None,
        "last_frontier_update": cache.last_frontier_update.isoformat() if cache.last_frontier_update else None,
    }


# ---------- Assets ----------

@app.get("/assets/summary")
def assets_summary():
    """טבלת מדדים לכל נכס."""
    try:
        cache.ensure_core_data()
        if cache.assets_summary is None:
            raise RuntimeError("Assets summary not ready")
        return cache.assets_summary.to_dict(orient="index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assets/cumulative")
def assets_cumulative(tail: int = 365):
    """
    צמיחה מצטברת לכל נכס.
    tail = מספר הימים האחרונים להחזיר (ברירת מחדל: 365).
    """
    try:
        cache.ensure_core_data()
        cumulative = cache.cumulative
        if cumulative is None:
            raise RuntimeError("Cumulative data not ready")

        if tail > 0:
            cumulative = cumulative.tail(tail)

        data = []
        for idx, row in cumulative.iterrows():
            entry = {"date": idx.strftime("%Y-%m-%d")}
            for col in cumulative.columns:
                entry[col] = float(row[col])
            data.append(entry)

        return {
            "meta": {
                "tail_days": tail,
                "tickers": list(cumulative.columns),
            },
            "data": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assets/correlation")
def assets_correlation():
    """מטריצת קורלציה בין הנכסים."""
    try:
        cache.ensure_core_data()
        if cache.daily_returns is None:
            raise RuntimeError("Daily returns not ready")

        corr = cache.daily_returns.corr().round(3)
        return corr.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Portfolios ----------

@app.get("/portfolios/summary")
def portfolios_summary():
    """טבלת מדדים ודירוגים לכל תיק."""
    try:
        cache.ensure_core_data()
        if cache.portfolio_summary is None:
            raise RuntimeError("Portfolio summary not ready")
        return cache.portfolio_summary.to_dict(orient="index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolios/cumulative")
def portfolios_cumulative(tail: int = 365):
    """צמיחה מצטברת של כל תיק."""
    try:
        cache.ensure_core_data()
        curves = cache.portfolio_cumulative
        if not curves:
            raise RuntimeError("Portfolio curves not ready")

        result = {}
        for name, series in curves.items():
            s = series.tail(tail) if tail > 0 else series
            result[name] = [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
                for idx, val in s.items()
            ]

        # מוסיף גם את התיקים האופטימליים אם קיימים
        for name, series in cache.optimal_curves.items():
            s = series.tail(tail) if tail > 0 else series
            result[name] = [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
                for idx, val in s.items()
            ]

        return {
            "meta": {
                "tail_days": tail,
            },
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolios/best")
def portfolios_best():
    """התיקים הטובים ביותר לפי מדדים שונים."""
    try:
        cache.ensure_core_data()
        if cache.best_portfolios is None:
            raise RuntimeError("Best portfolios not ready")
        return cache.best_portfolios
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Efficient Frontier ----------

@app.get("/frontier")
def frontier_points():
    """נקודות ה-Efficient Frontier (return, volatility, sharpe לכל תיק רנדומלי)."""
    try:
        cache.ensure_core_data()
        cache.ensure_frontier()
        if cache.frontier_points is None:
            raise RuntimeError("Frontier not ready")
        return cache.frontier_points
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/frontier/optimal")
def frontier_optimal():
    """משקולות וסטטיסטיקות של Max Sharpe & Min Volatility."""
    try:
        cache.ensure_core_data()
        cache.ensure_frontier()
        if cache.optimal_summary is None or cache.optimal_weights is None:
            raise RuntimeError("Optimal portfolios not ready")

        return {
            "summary": cache.optimal_summary.to_dict(orient="index"),
            "weights": cache.optimal_weights,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Snapshot חי ----------

@app.get("/live")
def live_snapshot():
    """
    Snapshot חי של 1 חודש אחרון עבור:
    SPY, QQQM, IWM, GC=F
    """
    try:
        tickers = ["SPY", "QQQM", "IWM", "GC=F"]
        final = {}

        for t in tickers:
            df = yf.download(t, period="1mo", auto_adjust=True, progress=False)
            if df is None or df.empty:
                continue

            df["returns"] = df["Close"].pct_change()
            final[t] = {
                "price": float(df["Close"].iloc[-1]),
                "daily_return": float(df["returns"].iloc[-1]),
                "mean_return": float(df["returns"].mean()),
                "volatility": float(df["returns"].std()),
            }

        if not final:
            raise RuntimeError("No data for live snapshot")

        return final
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
