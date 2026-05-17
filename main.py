# backend/main.py
"""
Trading Copilot — FastAPI Backend
Deploy en Render.com (free tier) o Railway.
"""
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client
import httpx
import websockets

load_dotenv()

from tools.market_tool import get_market_analysis, get_top_cryptos, get_current_price
from tools.news_tool import get_news
from tools.backtest_tool import run_backtest
from agents.trading_agent import build_agent, generate_trading_signal
from langchain_core.messages import HumanMessage

# ── App init ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Trading Copilot API",
    version="1.0.0",
    description="AI-powered crypto trading assistant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase client (service role — solo backend) ──────────────────────────
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
)

# ── WebSocket connections manager ──────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}  # symbol -> [ws]

    async def connect(self, ws: WebSocket, symbol: str):
        await ws.accept()
        self.active.setdefault(symbol, []).append(ws)

    def disconnect(self, ws: WebSocket, symbol: str):
        self.active.get(symbol, []).remove(ws)

    async def broadcast(self, symbol: str, data: dict):
        for ws in self.active.get(symbol, [])[:]:
            try:
                await ws.send_json(data)
            except Exception:
                self.active[symbol].remove(ws)

manager = ConnectionManager()


# ── Pydantic models ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: list[dict] = Field(default_factory=list)
    provider: str = "groq"

class BacktestRequest(BaseModel):
    symbol: str
    strategy: dict
    date_from: str
    date_to: str
    initial_capital: float = Field(default=1000.0, gt=0)
    timeframe: str = "1d"
    name: str = "My Strategy"

class SignalRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    provider: str = "groq"

class WatchlistRequest(BaseModel):
    symbol: str
    name: str
    market: str = "crypto"


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


# ── Market endpoints ───────────────────────────────────────────────────────
@app.get("/market/overview")
async def market_overview():
    """Top 10 cryptos con precios actuales."""
    try:
        data = await get_top_cryptos(limit=10)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/{symbol}")
async def market_detail(symbol: str, interval: str = "1h"):
    """Precio actual + indicadores técnicos + velas para un símbolo."""
    try:
        data = await get_market_analysis(symbol.upper(), interval)
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        return {"data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── News endpoints ─────────────────────────────────────────────────────────
@app.get("/news")
async def news_feed(symbol: Optional[str] = None, limit: int = 15):
    """Feed de noticias con sentimiento."""
    try:
        news = await get_news(symbol=symbol, limit=limit)
        return {"data": news, "count": len(news)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Signals endpoints ──────────────────────────────────────────────────────
@app.post("/signals/generate")
async def generate_signal(req: SignalRequest):
    """Genera una señal de trading para el símbolo dado."""
    try:
        signal = await generate_trading_signal(req.symbol, req.interval, req.provider)
        if "error" in signal:
            raise HTTPException(status_code=400, detail=signal["error"])

        # Guardar en Supabase (sin user_id = señal global)
        saved = supabase.table("signals").insert({
            **signal,
            "indicators":   json.dumps(signal.get("indicators", {})),
            "news_context": json.dumps(signal.get("news_context", [])),
        }).execute()

        return {"data": signal, "saved_id": saved.data[0]["id"] if saved.data else None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals")
async def get_signals(symbol: Optional[str] = None, limit: int = 20):
    """Historial de señales desde Supabase."""
    try:
        query = supabase.table("signals").select("*").order("created_at", desc=True).limit(limit)
        if symbol:
            query = query.eq("symbol", symbol.upper())
        result = query.execute()
        return {"data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Backtest endpoints ─────────────────────────────────────────────────────
@app.post("/backtest/run")
async def run_backtest_endpoint(req: BacktestRequest, background_tasks: BackgroundTasks):
    """Inicia un backtest. Retorna ID inmediatamente, resultado disponible después."""
    try:
        # Crear registro en Supabase con status PENDING
        record = supabase.table("backtests").insert({
            "name":           req.name,
            "symbol":         req.symbol.upper(),
            "strategy":       json.dumps(req.strategy),
            "timeframe":      req.timeframe,
            "date_from":      req.date_from,
            "date_to":        req.date_to,
            "initial_capital": req.initial_capital,
            "status":         "RUNNING",
        }).execute()

        backtest_id = record.data[0]["id"]

        # Ejecutar en background (no bloquea el request)
        background_tasks.add_task(
            _run_and_save_backtest,
            backtest_id, req.symbol, req.strategy,
            req.date_from, req.date_to, req.initial_capital, req.timeframe
        )

        return {"backtest_id": backtest_id, "status": "RUNNING"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_and_save_backtest(backtest_id, symbol, strategy, date_from, date_to, capital, timeframe):
    """Tarea de background: corre backtest y actualiza Supabase."""
    try:
        result = await run_backtest(symbol, strategy, date_from, date_to, capital, timeframe)
        supabase.table("backtests").update({
            "final_capital":    result.get("final_capital"),
            "total_return_pct": result.get("total_return_pct"),
            "win_rate":         result.get("win_rate"),
            "max_drawdown":     result.get("max_drawdown"),
            "sharpe_ratio":     result.get("sharpe_ratio"),
            "total_trades":     result.get("total_trades"),
            "winning_trades":   result.get("winning_trades"),
            "losing_trades":    result.get("losing_trades"),
            "trades":           json.dumps(result.get("trades", [])),
            "equity_curve":     json.dumps(result.get("equity_curve", [])),
            "status":           "DONE",
            "finished_at":      datetime.now(timezone.utc).isoformat(),
        }).eq("id", backtest_id).execute()
    except Exception as e:
        supabase.table("backtests").update({
            "status": "ERROR", "error_msg": str(e)
        }).eq("id", backtest_id).execute()


@app.get("/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """Obtiene resultado de un backtest por ID."""
    try:
        result = supabase.table("backtests").select("*").eq("id", backtest_id).single().execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return {"data": result.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
async def list_backtests(limit: int = 10):
    """Lista todos los backtests."""
    try:
        result = supabase.table("backtests").select(
            "id, name, symbol, status, total_return_pct, win_rate, total_trades, created_at"
        ).order("created_at", desc=True).limit(limit).execute()
        return {"data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat / Agent endpoint ──────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    """Envía un mensaje al agente LangGraph y retorna su respuesta."""
    try:
        agent = build_agent(provider=req.provider)

        # Reconstruir historial
        from langchain_core.messages import HumanMessage, AIMessage
        history_msgs = []
        for msg in req.history[-10:]:  # max 10 turnos de contexto
            if msg.get("role") == "user":
                history_msgs.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                history_msgs.append(AIMessage(content=msg["content"]))

        history_msgs.append(HumanMessage(content=req.message))

        result = await asyncio.to_thread(
            agent.invoke, {"messages": history_msgs}
        )

        last_msg = result["messages"][-1]
        response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # Persistir en Supabase si hay session_id
        if req.session_id:
            try:
                session = supabase.table("chat_sessions").select("messages").eq("id", req.session_id).single().execute()
                msgs = session.data.get("messages", []) if session.data else []
                msgs.append({"role": "user",      "content": req.message,      "timestamp": datetime.now(timezone.utc).isoformat()})
                msgs.append({"role": "assistant", "content": response_text,    "timestamp": datetime.now(timezone.utc).isoformat()})
                supabase.table("chat_sessions").update({"messages": json.dumps(msgs)}).eq("id", req.session_id).execute()
            except Exception:
                pass  # No fatal si falla persistencia

        return {"response": response_text, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket — precios en tiempo real via Binance ─────────────────────────
@app.websocket("/ws/prices/{symbol}")
async def ws_prices(websocket: WebSocket, symbol: str):
    """
    WebSocket que retransmite el stream de precios de Binance al cliente.
    Reconexión automática si Binance desconecta (cada 24h).
    """
    symbol_lower = symbol.lower() + "usdt"
    await manager.connect(websocket, symbol)

    async def stream_from_binance():
        uri = f"wss://stream.binance.com:9443/ws/{symbol_lower}@ticker"
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    async for raw in ws:
                        data = json.loads(raw)
                        payload = {
                            "symbol":     symbol.upper(),
                            "price":      float(data["c"]),
                            "change_24h": float(data["P"]),
                            "volume_24h": float(data["q"]),
                            "high_24h":   float(data["h"]),
                            "low_24h":    float(data["l"]),
                            "timestamp":  datetime.now(timezone.utc).isoformat(),
                        }
                        await manager.broadcast(symbol, payload)
            except Exception:
                await asyncio.sleep(5)  # espera y reconecta

    stream_task = asyncio.create_task(stream_from_binance())
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
        stream_task.cancel()