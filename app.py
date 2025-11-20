import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import urllib.parse
from collections import Counter
import re

# ==========================================
# 0. 頁面與 UI 設定 (升級版)
# ==========================================
st.set_page_config(page_title="Quant Arsenal v35 Pro", layout="wide", page_icon="🏯")

# 高級 CSS 美化 (Glassmorphism & Modern UI)
st.markdown("""
    <style>
    /* 全局字體與背景 */
    .main {
        background-color: #f4f6f9;
    }
    
    /* 標題優化 */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif; 
        color: #2c3e50; 
        font-weight: 700;
    }

    /* 卡片式設計 - 毛玻璃特效 */
    .stMetric {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* 定制 Tabs 樣式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 10px 10px 0 0;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.02);
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        border-top: 3px solid #3498db;
        color: #3498db !important;
        font-weight: bold;
    }

    /* 新聞卡片 */
    .news-card {
        padding: 15px; margin-bottom: 12px; border-radius: 10px; 
        background: #fff; border-left: 5px solid #ccc; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .news-card:hover { transform: translateX(5px); }
    .news-pos {border-left-color: #28a745;} 
    .news-neg {border-left-color: #dc3545;} 
    .news-neu {border-left-color: #6c757d;}
    
    /* 關鍵字標籤 */
    .keyword-tag {
        display: inline-block;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #1565c0;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 資料庫設定
# ==========================================
SECTOR_DB = {
    "🔮 量子電腦與未來科技": {'QUBT': 'Quantum Computing', 'IONQ': 'IonQ', 'QBTS': 'D-Wave', 'RGTI': 'Rigetti', 'SOUN': 'SoundHound'},
    "☁️ 雲端、AI 算力與資安": {'CRWV': 'CoreWeave', 'NBIS': 'Nebius', 'CRWD': 'CrowdStrike', 'PLTR': 'Palantir', 'PANW': 'Palo Alto', 'ZS': 'Zscaler', 'NET': 'Cloudflare', 'SMCI': 'Super Micro'},
    "₿ 加密貨幣與金融科技": {'CRCL': 'Circle (USDC)', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'COIN': 'Coinbase', 'MSTR': 'MicroStrategy', 'PYPL': 'PayPal', 'SQ': 'Block', 'HOOD': 'Robinhood'},
    "🇹🇼 台灣權值與熱門股": {'2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '0050.TW': '台灣50', '2603.TW': '長榮', '2382.TW': '廣達', '3008.TW': '大立光', '2881.TW': '富邦金'},
    "🇺🇸 S&P 500 重權股": {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'AMZN': 'Amazon', 'GOOGL': 'Alphabet', 'META': 'Meta', 'TSLA': 'Tesla', 'LLY': 'Eli Lilly'},
    "💊 醫療生技": {'LLY': 'Eli Lilly', 'NVO': 'Novo Nordisk', 'UNH': 'UnitedHealth', 'PFE': 'Pfizer', 'MRK': 'Merck', 'ISRG': 'Intuitive Surgical'},
    "📈 熱門 ETF": {'QQQ': 'Nasdaq 100', 'SPY': 'S&P 500', 'TLT': '20Y Bond', 'GLD': 'Gold', 'SMH': 'Semiconductor', '^VIX': 'VIX'}
}
ALL_STOCKS = {k: v for sector, stocks in SECTOR_DB.items() for k, v in stocks.items()}
BENCHMARKS = {'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', '0050.TW': '台灣 50', '^DJI': 'Dow Jones'}

CH_MAP = {
    "sector": "產業板塊", "industry": "細分行業", "fullTimeEmployees": "員工數",
    "returnOnEquity": "ROE (股東權益報酬率)", "profitMargins": "淨利率",
    "grossMargins": "毛利率", "trailingPE": "本益比 (PE)",
    "forwardPE": "預估本益比", "pegRatio": "PEG 指標",
    "priceToBook": "股價淨值比 (PB)", "debtToEquity": "負債權益比",
    "currentRatio": "流動比率", "freeCashflow": "自由現金流",
    "totalRevenue": "總營收", "marketCap": "市值"
}

FIN_LEXICON = {
    "bullish": ['surge', 'soar', 'jump', 'rally', 'record', 'high', 'beat', 'buy', 'growth', 'profit', 'outperform', 'up', 'gain', 'bull', 'strong', '大漲', '飆升', '新高', '利多', '強勢', '優於', '成長', '獲利', '買進', '強勁'],
    "bearish": ['plunge', 'drop', 'crash', 'fall', 'miss', 'loss', 'down', 'bear', 'sell', 'weak', 'cut', 'low', 'warn', 'slump', '重挫', '崩盤', '新低', '利空', '大跌', '不如', '虧損', '賣出', '警告', '砍', '疲軟']
}

# ==========================================
# 2. 核心運算 & 爬蟲
# ==========================================
@st.cache_data(ttl=3600)
def download_data(tickers, start_date, end_date):
    try:
        # 確保同時下載 Benchmark 數據以便快取，但不一定要合併到主 DataFrame
        df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df.empty: return pd.DataFrame()
        
        # 處理 MultiIndex Columns
        if 'Close' in df.columns: 
            df_close = df['Close']
        else:
            # 如果沒有顯式的 'Close'，嘗試處理不同版本的 yfinance 結構
            df_close = df.xs('Close', level=0, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
            
        # 如果只有一檔股票，Series 轉 DataFrame
        if isinstance(df_close, pd.Series): 
            df_close = df_close.to_frame(name=tickers[0])
            
        return df_close
    except Exception as e:
        st.error(f"數據下載錯誤: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_benchmark_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if 'Close' in df.columns: return df['Close']
        return df.iloc[:, 0] # Fallback
    except: return pd.Series()

@st.cache_data(ttl=3600)
def get_live_risk_free_rate():
    try: return yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1] / 100
    except: return 0.04

# --- 新聞爬蟲 ---
def get_google_news_rss(ticker):
    try:
        search_term = ticker.split('.')[0] 
        url = f"https://news.google.com/rss/search?q={search_term}+stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll('item')
        news_data = []
        for item in items[:15]:
            title = item.title.text
            news_data.append({"title": title, "link": item.link.text, "pubDate": item.pubDate.text, "source": "Google News"})
        return news_data
    except: return []

def calculate_financial_sentiment(text):
    text_lower = text.lower()
    score = 0
    bull_hits = sum(1 for w in FIN_LEXICON['bullish'] if w in text_lower)
    bear_hits = sum(1 for w in FIN_LEXICON['bearish'] if w in text_lower)
    if bull_hits > bear_hits: score = 0.6 + (0.1 * bull_hits)
    elif bear_hits > bull_hits: score = -0.6 - (0.1 * bear_hits)
    else: score = TextBlob(text).sentiment.polarity
    return max(min(score, 1.0), -1.0)

def analyze_sentiment_enhanced(ticker):
    news_data = get_google_news_rss(ticker)
    if not news_data:
        try:
            yf_news = yf.Ticker(ticker).news
            for n in yf_news[:5]:
                news_data.append({"title": n.get('title', ''), "link": n.get('link', ''), "pubDate": "Recent", "source": n.get('publisher', 'Yahoo')})
        except: pass
    
    if not news_data: return 0, [], {}, []

    scores = []
    all_text = ""
    for n in news_data:
        s = calculate_financial_sentiment(n['title'])
        n['score'] = s
        scores.append(s)
        all_text += n['title'] + " "
    
    avg_score = np.mean(scores)
    
    words = re.findall(r'\w+', all_text.lower())
    stop_words = set(['to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'the', 'a', 'and', 'is', 'stock', 'market', 'news', 'today', 'stocks', 'price', 'shares', 'company', 'corp', 'inc', 'limited', 'group', 'tw', 'us', '台積電', '股票', '台股', '美股', '市場', '報導', '表示', '指出', 'google', 'yahoo', 'cnbc', 'bloomberg', 'video'])
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1 and not w.isdigit()]
    top_keywords = Counter(filtered_words).most_common(10)
    
    stats = {
        "bull": sum(1 for s in scores if s > 0.1),
        "bear": sum(1 for s in scores if s < -0.1),
        "neutral": sum(1 for s in scores if -0.1 <= s <= 0.1)
    }
    
    return avg_score, news_data, stats, top_keywords

# --- 高階指標 ---
def calculate_advanced_metrics(series, rf_rate=0.04):
    if len(series) < 30: return None
    ret = series.pct_change().dropna()
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    ann_return = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ann_return - rf_rate) / ann_vol if ann_vol != 0 else 0
    neg_ret = ret[ret < 0]
    sortino = (ann_return - rf_rate) / (neg_ret.std() * np.sqrt(252)) if not neg_ret.empty and neg_ret.std() != 0 else 0
    cum_ret = (1 + ret).cumprod()
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    calmar = ann_return / abs(mdd) if mdd != 0 else 0
    win_rate = len(ret[ret > 0]) / len(ret)
    return {"Return": total_return, "Ann_Vol": ann_vol, "Sharpe": sharpe, "Sortino": sortino, "MDD": mdd, "Calmar": calmar, "Win_Rate": win_rate}

def calculate_ai_score(df_close, ticker, sentiment_score):
    score = 50
    try:
        prices = df_close[ticker].dropna()
        if len(prices) < 60: return 0, "N/A", {}
        ma20 = prices.rolling(20).mean().iloc[-1]
        ma60 = prices.rolling(60).mean().iloc[-1]
        if prices.iloc[-1] > ma20: score += 10
        if ma20 > ma60: score += 10
        m = calculate_advanced_metrics(prices)
        if m:
            if m['Sharpe'] > 1: score += 10
            if m['Sortino'] > 1.5: score += 5
            if m['MDD'] > -0.2: score += 5
        delta = prices.diff()
        rs = (delta.where(delta>0, 0).rolling(14).mean()) / (-delta.where(delta<0, 0).rolling(14).mean())
        rsi = 100 - (100/(1+rs))
        if rsi.iloc[-1] < 30: score += 15
        elif rsi.iloc[-1] > 75: score -= 10
        if sentiment_score > 0.1: score += 10
        elif sentiment_score < -0.1: score -= 10
        score = min(100, max(0, score))
        tag = "🚀 強力買進" if score >= 80 else "🟢 看多" if score >= 60 else "🔴 避險" if score <= 40 else "🟡 觀望"
        return score, tag, m
    except: return 0, "Error", {}

@st.cache_data(ttl=3600)
def get_company_profile_deep(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info
    except: return {}

# ==========================================
# 3. 回測引擎 (升級版：含 Benchmark)
# ==========================================
def calculate_max_drawdown_series(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

def backtest_engine(df_series, strategy_name, benchmark_series=None):
    data = pd.DataFrame(df_series)
    data.columns = ['Close']
    data['Return'] = data['Close'].pct_change()
    data['Signal'] = 0 

    # 策略邏輯
    if strategy_name == "💎 Buy & Hold": data['Signal'] = 1
    elif strategy_name == "🐯 Mark Minervini (Trend)":
        if len(data) > 200:
            data['MA50'] = data['Close'].rolling(50).mean(); data['MA150'] = data['Close'].rolling(150).mean(); data['MA200'] = data['Close'].rolling(200).mean()
            data['Signal'] = np.where((data['Close']>data['MA50']) & (data['MA50']>data['MA150']) & (data['MA150']>data['MA200']), 1, 0)
    elif strategy_name == "📈 布林通道突破":
        data['MA20'] = data['Close'].rolling(20).mean(); data['STD'] = data['Close'].rolling(20).std()
        data['Upper'] = data['MA20'] + (2*data['STD'])
        data['Signal'] = np.where(data['Close'] > data['Upper'], 1, 0)
    elif strategy_name == "🐢 海龜交易法則":
        data['High_20'] = data['Close'].shift(1).rolling(20).max()
        data['Low_10'] = data['Close'].shift(1).rolling(10).min()
        pos = 0; sigs = []
        for i in range(len(data)):
            if data['Close'].iloc[i] > data['High_20'].iloc[i]: pos = 1
            elif data['Close'].iloc[i] < data['Low_10'].iloc[i]: pos = 0
            sigs.append(pos)
        data['Signal'] = sigs
    elif strategy_name == "⚔️ MA 黃金交叉":
        data['Signal'] = np.where(data['Close'].rolling(20).mean() > data['Close'].rolling(60).mean(), 1, 0)
    elif strategy_name == "⚡ RSI 極限反轉":
        delta = data['Close'].diff()
        rs = (delta.where(delta>0,0).rolling(14).mean())/(-delta.where(delta<0,0).rolling(14).mean())
        rsi = 100-(100/(1+rs))
        data['Signal'] = np.where(rsi<30, 1, np.where(rsi>70, 0, 1))

    data['Strategy_Ret'] = data['Signal'].shift(1) * data['Return']
    data['Buy_Hold_Ret'] = data['Return']
    
    # Benchmark 處理
    if benchmark_series is not None:
        # 對齊索引
        bench_aligned = benchmark_series.reindex(data.index).fillna(method='ffill').pct_change()
        data['Benchmark_Ret'] = bench_aligned
    
    data = data.dropna()
    data['Cum_Strategy'] = (1 + data['Strategy_Ret']).cumprod()
    data['Cum_Buy_Hold'] = (1 + data['Buy_Hold_Ret']).cumprod()
    
    if benchmark_series is not None and 'Benchmark_Ret' in data.columns:
        data['Cum_Benchmark'] = (1 + data['Benchmark_Ret']).cumprod()
    
    data['Drawdown'] = calculate_max_drawdown_series(data['Cum_Strategy'])
    
    data['Position_Change'] = data['Signal'].diff()
    data['Buy_Signal_Price'] = np.where(data['Position_Change'] == 1, data['Close'], np.nan)
    data['Sell_Signal_Price'] = np.where(data['Position_Change'] == -1, data['Close'], np.nan)
    return data

def monte_carlo_simulation_v3(df_close, days_forecast=252, iterations=200):
    try:
        series = df_close.iloc[:, 0] if isinstance(df_close, pd.DataFrame) else df_close
        last_price = series.iloc[-1]
        log_returns = np.log(1 + series.pct_change().dropna())
        u = log_returns.mean(); var = log_returns.var()
        drift = u - (0.5 * var); stdev = log_returns.std()
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days_forecast, iterations)))
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = last_price
        for t in range(1, days_forecast): price_paths[t] = price_paths[t-1] * daily_returns[t]
        final_prices = price_paths[-1]
        stats = {
            "P10 (悲觀)": np.percentile(final_prices, 10),
            "P50 (中立)": np.percentile(final_prices, 50),
            "P90 (樂觀)": np.percentile(final_prices, 90),
            "獲利機率": len(final_prices[final_prices > last_price]) / iterations
        }
        return price_paths, final_prices, stats
    except: return None, None, None

# ==========================================
# 4. UI 主程式
# ==========================================
st.sidebar.markdown("### 👨‍🎓 國立陽明交通大學 / 管理科學系")
st.sidebar.caption("Financial Data Analysis Project Pro")
st.sidebar.markdown("---")

st.sidebar.title("🏯 Quant Arsenal v35")
rf_rate = get_live_risk_free_rate()
st.sidebar.metric("無風險利率 (Risk Free)", f"{rf_rate:.2%}")

selected_sector = st.sidebar.selectbox("🔍 板塊篩選:", ["(所有資產)"] + list(SECTOR_DB.keys()))
current_options = [f"{k} - {v}" for k, v in ALL_STOCKS.items()] if selected_sector == "(所有資產)" else [f"{k} - {v}" for k, v in SECTOR_DB[selected_sector].items()]

default_tickers = ["NVDA - NVIDIA", "TSLA - Tesla"]
sel = st.sidebar.multiselect("選擇標的:", current_options, default=[x for x in default_tickers if x in current_options])
selected_tickers = [x.split(" - ")[0] for x in sel]

st.sidebar.subheader("📅 全局設定")
# Benchmark Selector
benchmark_ticker = st.sidebar.selectbox("比較基準 (Benchmark):", list(BENCHMARKS.keys()), format_func=lambda x: f"{x} - {BENCHMARKS[x]}")
start_date = st.sidebar.date_input("起始日", value=datetime.today() - timedelta(days=365*3))
end_date = st.sidebar.date_input("結束日", value=datetime.today())
run_btn = st.sidebar.button("🚀 執行全量分析", type="primary")

st.title("📊 金融市場數據分析終端 (Quant Arsenal Pro)")

if run_btn or selected_tickers:
    if not selected_tickers: st.warning("請選擇資產")
    else:
        with st.spinner(f'📡 正在連線全球市場數據庫...'):
            # 同時下載 股票 與 基準
            df_close = download_data(selected_tickers, start_date, end_date)
            df_benchmark = get_benchmark_data(benchmark_ticker, start_date, end_date)
        
        if df_close.empty: st.error("❌ 數據下載失敗")
        else:
            valid_tickers = [t for t in selected_tickers if t in df_close.columns]
            df_close = df_close[valid_tickers]
            
            # Toast Notification
            st.toast(f"分析完成！已載入 {len(valid_tickers)} 檔資產與基準 {benchmark_ticker}", icon="✅")
            
            # Dashboard
            st.subheader("🏆 AI 投資儀表板")
            rank_data = []
            progress_bar = st.progress(0)
            for idx, t in enumerate(valid_tickers):
                s_score, _, _, _ = analyze_sentiment_enhanced(t)
                score, tag, m = calculate_ai_score(df_close, t, s_score)
                if m:
                    rank_data.append({
                        "代號": t, "推薦指數": score, "評級": tag,
                        "Sharpe": m['Sharpe'], "Sortino": m['Sortino'],
                        "MDD": m['MDD'], "波動率": m['Ann_Vol'], "情緒": s_score
                    })
                progress_bar.progress((idx + 1) / len(valid_tickers))
            progress_bar.empty()
            
            rank_df = pd.DataFrame(rank_data).sort_values("推薦指數", ascending=False)
            st.dataframe(rank_df.style.background_gradient(subset=['推薦指數'], cmap='RdYlGn', vmin=0, vmax=100).format("{:.2f}", subset=['Sharpe', 'Sortino', '情緒']).format("{:.2%}", subset=['MDD', '波動率']), use_container_width=True, hide_index=True)

            # Tabs
            tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📈 股價走勢", "🧠 策略回測", "🏢 公司簡介", "⚖️ 資產配置 (Pro)", "🔮 蒙地卡羅", "📊 相關性", "📰 新聞情緒", "📚 投資百科"
            ])

            # Tab 0: Price Chart
            with tab0:
                st.subheader("📈 歷史股價走勢比較")
                chart_mode = st.radio("顯示模式:", ["相對漲跌幅 (%)", "絕對價格"], horizontal=True)
                plot_df = (df_close / df_close.iloc[0] - 1) if chart_mode == "相對漲跌幅 (%)" else df_close
                y_fmt = ".2%" if chart_mode == "相對漲跌幅 (%)" else ".2f"
                fig_price = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title=f"多股走勢圖")
                fig_price.update_layout(hovermode="x unified", yaxis_tickformat=y_fmt, template="plotly_white")
                st.plotly_chart(fig_price, use_container_width=True)

            # Tab 1: Backtest (Upgraded with Benchmark)
            with tab1:
                c1, c2 = st.columns(2)
                bt_target = c1.selectbox("回測標的:", valid_tickers, key='bt')
                strategy = c2.selectbox("選擇策略:", ["💎 Buy & Hold", "🐯 Mark Minervini (Trend)", "📈 布林通道突破", "🐢 海龜交易法則", "⚔️ MA 黃金交叉", "⚡ RSI 極限反轉"])
                
                # 傳入 Benchmark 數據
                bt_df = backtest_engine(df_close[bt_target], strategy, df_benchmark)
                
                if not bt_df.empty:
                    total_days = (bt_df.index[-1] - bt_df.index[0]).days
                    years = total_days / 365.25
                    cagr = (bt_df['Cum_Strategy'].iloc[-1])**(1/years) - 1 if years > 0 else 0
                    
                    # Benchmark CAGR
                    bench_cagr = 0
                    has_bench = 'Cum_Benchmark' in bt_df.columns
                    if has_bench:
                        bench_cagr = (bt_df['Cum_Benchmark'].iloc[-1])**(1/years) - 1 if years > 0 else 0
                        alpha = cagr - bench_cagr
                    else:
                        alpha = 0
                        
                    alpha_desc = "🔥 跑贏大盤" if alpha > 0 else "❄️ 落後大盤"
                    mdd = bt_df['Drawdown'].min()
                    wins = bt_df[bt_df['Strategy_Ret'] > 0]
                    win_rate = len(wins) / len(bt_df) if len(bt_df) > 0 else 0

                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("策略年化 (CAGR)", f"{cagr:.2%}")
                    k2.metric("基準年化 (Benchmark)", f"{bench_cagr:.2%}", help=f"Benchmark: {BENCHMARKS.get(benchmark_ticker, benchmark_ticker)}")
                    k3.metric("Alpha (超額報酬)", f"{alpha:.2%}", delta=alpha_desc)
                    k4.metric("最大回落 (MDD)", f"{mdd:.2%}", delta_color="inverse")
                    k5.metric("日勝率", f"{win_rate:.1%}")
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                    # 策略線
                    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Strategy'], name='策略淨值', fill='tozeroy', line=dict(color='#2ecc71', width=2)), row=1, col=1)
                    # 基準線
                    if has_bench:
                        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cum_Benchmark'], name=f'基準 ({benchmark_ticker})', line=dict(color='#95a5a6', dash='dash', width=2)), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Buy_Signal_Price'] * bt_df['Cum_Strategy'] / bt_df['Close'], mode='markers', name='買進', marker=dict(color='#3498db', size=10, symbol='triangle-up')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Sell_Signal_Price'] * bt_df['Cum_Strategy'] / bt_df['Close'], mode='markers', name='賣出', marker=dict(color='#e74c3c', size=10, symbol='triangle-down')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Drawdown'], name='回落幅度', fill='tozeroy', line=dict(color='#e74c3c', width=1)), row=2, col=1)
                    fig.update_layout(template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 2: Company Profile
            with tab2:
                t_fund = st.selectbox("選擇公司:", valid_tickers)
                with st.spinner("獲取公司深度資料..."):
                    info = get_company_profile_deep(t_fund)
                if info:
                    summary_text = info.get('longBusinessSummary', '')
                    translate_url = f"https://translate.google.com/?sl=en&tl=zh-TW&text={urllib.parse.quote(summary_text)}&op=translate"
                    col_a, col_b = st.columns([3, 1])
                    with col_a: st.subheader(f"🏢 {info.get('shortName', t_fund)}")
                    with col_b: st.markdown(f"[🌐 一鍵翻譯簡介]({translate_url})", unsafe_allow_html=True)
                    st.info(f"**簡介摘要:** {summary_text[:500]}... (點擊上方按鈕查看中文)")
                    st.markdown("#### 📊 核心財務數據")
                    c1, c2, c3, c4 = st.columns(4)
                    keys_map = ["sector", "industry", "fullTimeEmployees", "marketCap", "returnOnEquity", "profitMargins", "trailingPE", "pegRatio", "debtToEquity", "currentRatio", "freeCashflow", "totalRevenue"]
                    for i, k in enumerate(keys_map):
                        col = [c1, c2, c3, c4][i % 4]
                        label = CH_MAP.get(k, k)
                        val = info.get(k)
                        if val is None: s_val = "N/A"
                        elif isinstance(val, str): s_val = val
                        elif "Margins" in k or "return" in k: s_val = f"{val:.2%}"
                        elif "Cap" in k or "Revenue" in k or "Cash" in k: s_val = f"${val/1e9:.2f} B"
                        else: s_val = f"{val:.2f}"
                        col.metric(label, s_val)

            # Tab 3: Asset Allocation (Vectorized Optimization)
            with tab3:
                if len(valid_tickers) > 1:
                    st.subheader("⚖️ 效率前緣與風險配置 (High-Performance Optimized)")
                    ret = df_close.pct_change().dropna()
                    mean_ret = ret.mean() * 252
                    cov = ret.cov() * 252
                    
                    num_ports = 10000 # 提升到 10,000 次模擬
                    num_assets = len(valid_tickers)
                    
                    # --- 核心優化：向量化運算 ---
                    # 1. 一次生成所有權重矩陣 (N x Assets)
                    weights = np.random.random((num_ports, num_assets))
                    weights /= np.sum(weights, axis=1)[:, np.newaxis] # 正規化
                    
                    # 2. 矩陣乘法計算報酬 (N x 1)
                    port_rets = np.dot(weights, mean_ret)
                    
                    # 3. 向量化計算波動率 (N x 1)
                    # Variance = diag(w @ Cov @ w.T) -> 但這樣會生成 N x N 矩陣爆記憶體
                    # Optimized: sum((w @ Cov) * w, axis=1)
                    port_vols = np.sqrt(np.sum(np.dot(weights, cov) * weights, axis=1))
                    
                    # 4. 計算 Sharpe
                    port_sharpes = (port_rets - rf_rate) / port_vols
                    
                    # 5. 找出最佳點
                    max_sharpe_idx = np.argmax(port_sharpes)
                    min_vol_idx = np.argmin(port_vols)
                    
                    ms_w = weights[max_sharpe_idx]
                    mv_w = weights[min_vol_idx]
                    # ---------------------------
                    
                    fig_ef = go.Figure()
                    # 使用 Scattergl 加速大量點渲染
                    fig_ef.add_trace(go.Scattergl(x=port_vols, y=port_rets, mode='markers', marker=dict(color=port_sharpes, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe"), opacity=0.6, size=4), name='隨機組合'))
                    fig_ef.add_trace(go.Scatter(x=[port_vols[max_sharpe_idx]], y=[port_rets[max_sharpe_idx]], mode='markers', marker=dict(color='#e74c3c', size=18, symbol='star', line=dict(width=2, color='white')), name='★ 最佳夏普 (Max Sharpe)'))
                    fig_ef.add_trace(go.Scatter(x=[port_vols[min_vol_idx]], y=[port_rets[min_vol_idx]], mode='markers', marker=dict(color='#3498db', size=18, symbol='diamond', line=dict(width=2, color='white')), name='♦ 最小波動 (Min Vol)'))
                    fig_ef.update_layout(title=f"效率前緣 (模擬 {num_ports:,} 種組合)", xaxis_title="風險 (年化波動率)", yaxis_title="預期報酬", template="plotly_white", height=600)
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### 🔥 積極型：最佳夏普配置")
                        st.plotly_chart(px.pie(values=ms_w, names=valid_tickers, title=f"Sharpe: {port_sharpes[max_sharpe_idx]:.2f}", hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold), use_container_width=True)
                    with c2:
                        st.markdown("#### 🛡️ 防禦型：最小波動配置")
                        st.plotly_chart(px.pie(values=mv_w, names=valid_tickers, title=f"Volatility: {port_vols[min_vol_idx]:.2%}", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
                else: st.warning("資產配置需要至少兩檔股票。")

            # Tab 4: Monte Carlo
            with tab4:
                mc_t = st.selectbox("模擬標的:", valid_tickers, key='mc')
                paths, finals, stats = monte_carlo_simulation_v3(df_close[mc_t])
                if paths is not None:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("當前股價", f"${paths[0,0]:.2f}")
                    c2.metric("P50 中立預測", f"${stats['P50 (中立)']:.2f}")
                    c3.metric("P90 樂觀預測", f"${stats['P90 (樂觀)']:.2f}")
                    c4.metric("獲利機率", f"{stats['獲利機率']:.1%}")
                    fig_path = go.Figure()
                    # 限制顯示路徑數量以提升效能
                    for i in range(min(50, paths.shape[1])): fig_path.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1), opacity=0.1, showlegend=False, hoverinfo='skip'))
                    fig_path.add_trace(go.Scatter(y=paths.mean(axis=1), mode='lines', name='平均路徑', line=dict(color='#e74c3c', width=3)))
                    st.plotly_chart(fig_path, use_container_width=True)
                    fig_hist = px.histogram(finals, nbins=30, title="一年後價格分布機率", color_discrete_sequence=['#3498db'])
                    fig_hist.add_vline(x=paths[0,0], line_dash="dash", line_color="green", annotation_text="Current")
                    st.plotly_chart(fig_hist, use_container_width=True)

            # Tab 5: Correlation
            with tab5:
                if len(valid_tickers)>1:
                    fig = px.imshow(df_close.pct_change().corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 6: News (Visual Upgrade)
            with tab6:
                news_t = st.selectbox("新聞標的:", valid_tickers, key='news')
                score, news_data, stats, keywords = analyze_sentiment_enhanced(news_t)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.subheader("情緒分佈")
                    fig_pie = px.pie(names=['看多 (Bull)', '看空 (Bear)', '中立 (Neutral)'], values=[stats['bull'], stats['bear'], stats['neutral']], hole=0.5, color_discrete_sequence=['#28a745', '#dc3545', '#6c757d'])
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown("#### 熱門關鍵字")
                    for word, count in keywords:
                        st.markdown(f"<span class='keyword-tag'>{word} ({count})</span>", unsafe_allow_html=True)

                with c2:
                    st.subheader("最新新聞快訊")
                    for n in news_data:
                        css = "news-pos" if n['score']>0.1 else "news-neg" if n['score']<-0.1 else "news-neu"
                        st.markdown(f"""<div class="news-card {css}"><a href="{n['link']}" target="_blank" style="text-decoration:none; color:#333;"><b>{n['title']}</b></a><br><small style="color:#666;">{n['pubDate']} | {n['source']} | Score: {n['score']:.2f}</small></div>""", unsafe_allow_html=True)

            # Tab 7: Encyclopedia
            with tab7:
                st.markdown("## 📚 金融知識百科 (Management Science)")
                with st.expander("🧠 基礎指標 (Basic Metrics)", expanded=True):
                    st.markdown("""
                    * **Alpha (α)**: 投資組合的超額回報。正值代表跑贏大盤 (Benchmark)。
                    * **Beta (β)**: 衡量相對於大盤的波動性。
                    * **CAGR**: 年化複合成長率。
                    * **Correlation**: 資產間的連動程度。
                    """)
                with st.expander("⚖️ 風險指標 (Risk Metrics)", expanded=True):
                    st.markdown("""
                    * **Sharpe Ratio**: 單位總風險的超額報酬。
                    * **Sortino Ratio**: 只考慮下跌風險的夏普值。
                    * **MDD**: 最大回落，資產從高點跌到低點的最大幅度。
                    """)
                with st.expander("🛠️ 策略與技術 (Strategy)"):
                    st.markdown("""
                    * **VCP (Mark Minervini)**: 波動率收縮型態，尋找趨勢發動點。
                    * **Bollinger Bands**: 布林通道，利用標準差判斷超買超賣。
                    """)
                with st.expander("📊 投資組合理論 (MPT)"):
                    st.markdown("""
                    * **Efficient Frontier**: 效率前緣，在相同風險下提供最高報酬的組合。
                    """)