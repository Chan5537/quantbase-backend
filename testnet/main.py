from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import requests, datetime, json, os
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.transaction import Transaction
from solders.system_program import TransferParams, transfer

# --- Solana setup ---
client = Client("https://api.devnet.solana.com")
kp_path = os.path.expanduser("~/.config/solana/id.json")
with open(kp_path) as f:
    secret = bytes(json.load(f))
kp = Keypair.from_bytes(secret)
pub = kp.pubkey()

# --- Price source ---
url = "https://api.coingecko.com/api/v3/simple/price"
params = {"ids": "solana", "vs_currencies": "usd"}

# --- Dash UI ---
app = Dash(__name__)
app.layout = html.Div([
    html.H2("SOL/USD Live Bot — Buy Low, Sell High (Devnet)"),
    dcc.Graph(id="graph", style={"height": "70vh"}),
    html.Div(id="wallet-info", style={"fontSize": "20px", "marginTop": "20px"}),
    html.Div(id="trade-log", style={"fontSize": "16px", "marginTop": "10px", "color": "#00ff99"}),
    dcc.Interval(id="tick", interval=12_000, n_intervals=0)  # ~5 calls/min
])

# --- Globals ---
history, last_price, trade_log = [], None, []

@app.callback(
    Output("graph", "figure"),
    Output("wallet-info", "children"),
    Output("trade-log", "children"),
    Input("tick", "n_intervals")
)
def update(_):
    global last_price
    price = None
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        price = float(r.json()["solana"]["usd"])
        history.append((datetime.datetime.now(), price))
    except Exception as e:
        print("Price fetch error:", e)

    # --- Fetch Devnet balance ---
    try:
        lamports = client.get_balance(pub).value
        sol_balance = lamports / 1e9
    except Exception as e:
        print("Balance fetch error:", e)
        sol_balance = 0.0

    # --- Trading logic ---
    if last_price is not None and price is not None:
        dest = Keypair().pubkey()
        tx = Transaction()
        if price < last_price:
            # BUY (simulate transfer in)
            tx.add(transfer(TransferParams(from_pubkey=pub, to_pubkey=dest, lamports=10_000)))
            action = "BUY"
        elif price > last_price:
            # SELL (simulate transfer out)
            tx.add(transfer(TransferParams(from_pubkey=pub, to_pubkey=dest, lamports=10_000)))
            action = "SELL"
        else:
            action = None

        if action:
            try:
                sig = client.send_transaction(tx, kp)
                trade_log.append(f"{action} 0.00001 SOL @ ${price:.2f} (Tx: {sig.value[:8]}...)")
            except Exception as e:
                trade_log.append(f"{action} failed @ ${price:.2f}: {e}")

    last_price = price

    # --- Graph ---
    if not history:
        return go.Figure(), "Loading...", "No trades yet."
    times, prices = zip(*history[-200:])
    fig = go.Figure([go.Scatter(x=times, y=prices, mode="lines+markers", line=dict(color="royalblue"))])
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="SOL/USD ($)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # --- Wallet + Trades display ---
    if price is not None:
        usd_value = sol_balance * price
        info = f"Wallet: {pub}\nBalance: {sol_balance:.4f} SOL ≈ ${usd_value:,.2f} @ ${price:.2f}/SOL"
    else:
        info = "Loading wallet data..."
    trades = "<br>".join(trade_log[-6:]) if trade_log else "No trades yet."

    return fig, info, trades

if __name__ == "__main__":
    app.run(debug=False)
