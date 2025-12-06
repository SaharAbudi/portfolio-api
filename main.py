from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI()

# מאפשר ל-React שלך להתחבר ל-API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # אפשר לשים רק את הדומיין שלך אחרי שיפרס
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/live")
def live_portfolio():
    tickers = ["SPY", "QQQM", "IWM", "GC=F"]
    final = {}

    for t in tickers:
        df = yf.download(t, period="1mo", auto_adjust=True)
        df["returns"] = df["Close"].pct_change()

        final[t] = {
            "price": float(df["Close"].iloc[-1]),
            "daily_return": float(df["returns"].iloc[-1]),
            "mean_return": float(df["returns"].mean()),
            "volatility": float(df["returns"].std())
        }

    return final
