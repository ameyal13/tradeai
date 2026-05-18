# backend/main.py
"""
Trading Copilot — FastAPI Backend
Deploy en Render.com (free tier) o Railway.
"""
import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from supabase import create_client, Client
import httpx
import websockets

load_dotenv()

from tools.market_tool import get_market_analysis, get_top_cryptos, get_current_price
from tools.news_tool import get_news
from tools.backtest_tool import run_backtest
from tools.historical_data import fetch_binance_klines, normalize_ohlcv as normalize_historical_ohlcv
from tools.historical_replay import run_historical_replay
from tools.prediction_journal import (
    EVALUATED,
    INVALID,
    PredictionStore,
    evaluate_prediction_against_candles,
    fetch_future_klines,
    metrics_by_signal,
    metrics_by_strategy,
    metrics_by_strategy_mode,
    metrics_by_symbol_timeframe,
    normalize_prediction,
    parse_dt,
    prediction_payload_from_signal_response,
    utc_now,
)
from tools.strategy_optimizer import run_walk_forward_optimizer
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
supabase: Optional[Client] = None
if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
    try:
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        )
    except Exception:
        supabase = None
prediction_store = PredictionStore(supabase)

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
    commission_pct: float = Field(default=0.001, ge=0)
    slippage_pct: float = Field(default=0.0005, ge=0)
    spread_pct: float = Field(default=0.0003, ge=0)
    risk_per_trade_pct: float = Field(default=0.01, gt=0)
    min_volume: Optional[float] = Field(default=None, ge=0)

class SignalRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    provider: str = "groq"
    strategy_mode: str = "deterministic"
    horizon_minutes: int = Field(default=60, gt=0)

class PredictionCreateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    user_id: Optional[str] = None
    symbol: str
    timeframe: str = "1h"
    strategy_mode: str = "deterministic"
    strategy_name: str = "manual"
    strategy_version: str = "v1"
    signal: str
    confidence: float = Field(ge=0, le=100)
    entry_price: float = Field(gt=0)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    horizon_minutes: int = Field(default=60, gt=0)
    input_features: dict = Field(default_factory=dict)
    reasoning: str = ""
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    created_at: Optional[str] = None

class PredictionEvaluateRequest(BaseModel):
    prediction_id: str
    candles: list[dict] = Field(default_factory=list)
    commission_pct: float = Field(default=0.001, ge=0)
    slippage_pct: float = Field(default=0.0005, ge=0)
    spread_pct: float = Field(default=0.0003, ge=0)

class WatchlistRequest(BaseModel):
    symbol: str
    name: str
    market: str = "crypto"

class ReplayRunRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    strategy_mode: str = "deterministic"
    candles: list[dict] = Field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    limit: int = Field(default=300, gt=0)
    horizon_candles: int = Field(default=3, gt=0)
    horizon_minutes: int = Field(default=60, gt=0)
    commission_pct: float = Field(default=0.001, ge=0)
    slippage_pct: float = Field(default=0.0005, ge=0)
    spread_pct: float = Field(default=0.0003, ge=0)
    step_size: int = Field(default=1, gt=0)
    min_history: int = Field(default=50, gt=1)
    max_predictions: Optional[int] = Field(default=None, gt=0)
    strategy_params: dict = Field(default_factory=dict)
    persist: bool = False

class OptimizerRunRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    strategy_mode: str = "deterministic"
    candles: list[dict] = Field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    limit: int = Field(default=300, gt=0)
    train_size: int = Field(default=90, gt=10)
    validation_size: int = Field(default=45, gt=10)
    horizon_candles: int = Field(default=3, gt=0)
    min_history: int = Field(default=40, gt=1)
    step_size: int = Field(default=5, gt=0)
    parameter_grid: Optional[dict] = None


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
        signal = await generate_trading_signal(
            req.symbol,
            req.interval,
            req.provider,
            strategy_mode=req.strategy_mode,
            horizon_minutes=req.horizon_minutes,
        )
        if "error" in signal:
            raise HTTPException(status_code=400, detail=signal["error"])

        saved_id = None
        try:
            if supabase is not None:
                saved = supabase.table("signals").insert({
                    **signal,
                    "indicators":   json.dumps(signal.get("indicators", {})),
                    "news_context": json.dumps(signal.get("news_context", [])),
                }).execute()
                saved_id = saved.data[0]["id"] if saved.data else None
        except Exception:
            pass

        journal_entry = prediction_store.create_prediction(
            prediction_payload_from_signal_response(
                signal,
                timeframe=req.interval,
                requested_strategy_mode=req.strategy_mode,
                requested_horizon_minutes=req.horizon_minutes,
                provider=req.provider,
            )
        )

        return {"data": signal, "saved_id": saved_id, "prediction_id": journal_entry["id"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals")
async def get_signals(symbol: Optional[str] = None, limit: int = 20):
    """Historial de señales desde Supabase."""
    try:
        if supabase is None:
            return {"data": []}
        query = supabase.table("signals").select("*").order("created_at", desc=True).limit(limit)
        if symbol:
            query = query.eq("symbol", symbol.upper())
        result = query.execute()
        return {"data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Prediction journal / outcome evaluator ─────────────────────────────────
@app.post("/predictions/create")
async def create_prediction(req: PredictionCreateRequest):
    """Crea una predicción medible en el prediction journal."""
    try:
        prediction = prediction_store.create_prediction(req.model_dump(exclude_none=True))
        return {"data": prediction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
async def list_predictions(status: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100):
    """Lista predicciones del journal."""
    try:
        return {"data": prediction_store.list_predictions(status=status, symbol=symbol, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Obtiene una predicción por ID."""
    prediction = prediction_store.get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"data": prediction}


async def _evaluate_prediction(prediction: dict, req: Optional[PredictionEvaluateRequest] = None) -> dict:
    start = parse_dt(prediction["created_at"])
    end = start + timedelta(minutes=int(prediction["horizon_minutes"]))
    if req and req.candles:
        candles = pd.DataFrame(req.candles)
        commission_pct = req.commission_pct
        slippage_pct = req.slippage_pct
        spread_pct = req.spread_pct
    else:
        candles = await fetch_future_klines(prediction["symbol"], prediction["timeframe"], start, end)
        commission_pct = 0.001
        slippage_pct = 0.0005
        spread_pct = 0.0003

    outcome = evaluate_prediction_against_candles(
        prediction,
        candles,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        spread_pct=spread_pct,
    )
    saved_outcome = prediction_store.create_outcome(outcome)
    status = INVALID if outcome["outcome"] == "INVALID_DATA" else EVALUATED
    prediction_store.update_prediction_status(prediction["id"], status)
    return saved_outcome


@app.post("/predictions/evaluate")
async def evaluate_prediction(req: PredictionEvaluateRequest):
    """Evalúa una predicción contra velas posteriores a su timestamp."""
    prediction = prediction_store.get_prediction(req.prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    try:
        return {"data": await _evaluate_prediction(prediction, req)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictions/evaluate-due")
async def evaluate_due_predictions(limit: int = 100):
    """Evalúa predicciones pending cuyo horizonte ya expiró."""
    evaluated = []
    errors = []
    for prediction in prediction_store.due_predictions(now=utc_now(), limit=limit):
        try:
            evaluated.append(await _evaluate_prediction(prediction))
        except Exception as e:
            errors.append({"prediction_id": prediction.get("id"), "error": str(e)})
    return {"evaluated": evaluated, "errors": errors, "count": len(evaluated)}


@app.get("/metrics/signals")
async def get_signal_metrics():
    """Métricas agrupadas por BUY/SELL/HOLD."""
    predictions = prediction_store.list_predictions(limit=10000)
    outcomes = prediction_store.list_outcomes()
    return {"data": metrics_by_signal(predictions, outcomes)}


@app.get("/metrics/strategies")
async def get_strategy_metrics():
    """Métricas agrupadas por estrategia."""
    predictions = prediction_store.list_predictions(limit=10000)
    outcomes = prediction_store.list_outcomes()
    return {"data": metrics_by_strategy(predictions, outcomes)}


@app.get("/metrics/strategy-modes")
async def get_strategy_mode_metrics():
    """Métricas agrupadas por deterministic/model_based/hybrid."""
    predictions = prediction_store.list_predictions(limit=10000)
    outcomes = prediction_store.list_outcomes()
    return {"data": metrics_by_strategy_mode(predictions, outcomes)}


@app.get("/metrics/symbol-timeframes")
async def get_symbol_timeframe_metrics():
    """Métricas agrupadas por symbol/timeframe."""
    predictions = prediction_store.list_predictions(limit=10000)
    outcomes = prediction_store.list_outcomes()
    return {"data": metrics_by_symbol_timeframe(predictions, outcomes)}


# ── Historical replay / optimizer ──────────────────────────────────────────
async def _candles_from_request(req):
    if req.candles:
        return normalize_historical_ohlcv(pd.DataFrame(req.candles))
    return await fetch_binance_klines(
        req.symbol,
        req.interval,
        start_time=req.start_time,
        end_time=req.end_time,
        limit=req.limit,
    )


@app.post("/replay/run")
async def replay_run(req: ReplayRunRequest):
    """Run historical replay with manual candles or downloaded Binance candles."""
    try:
        candles = await _candles_from_request(req)
        result = run_historical_replay(
            candles,
            symbol=req.symbol,
            timeframe=req.interval,
            strategy_mode=req.strategy_mode,
            horizon_candles=req.horizon_candles,
            horizon_minutes=req.horizon_minutes,
            commission_pct=req.commission_pct,
            slippage_pct=req.slippage_pct,
            spread_pct=req.spread_pct,
            step_size=req.step_size,
            min_history=req.min_history,
            max_predictions=req.max_predictions,
            strategy_params=req.strategy_params,
            store=prediction_store if req.persist else None,
        )
        return {"data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimizer/run")
async def optimizer_run(req: OptimizerRunRequest):
    """Run walk-forward parameter optimization."""
    try:
        candles = await _candles_from_request(req)
        result = run_walk_forward_optimizer(
            candles,
            symbol=req.symbol,
            timeframe=req.interval,
            strategy_mode=req.strategy_mode,
            train_size=req.train_size,
            validation_size=req.validation_size,
            horizon_candles=req.horizon_candles,
            min_history=req.min_history,
            step_size=req.step_size,
            parameter_grid=req.parameter_grid,
        )
        return {"data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Backtest endpoints ─────────────────────────────────────────────────────
@app.post("/backtest/run")
async def run_backtest_endpoint(req: BacktestRequest, background_tasks: BackgroundTasks):
    """Inicia un backtest V2. Retorna ID inmediatamente, resultado disponible después."""
    try:
        strategy = {
            **req.strategy,
            "commission_pct": req.commission_pct,
            "slippage_pct": req.slippage_pct,
            "spread_pct": req.spread_pct,
            "risk_per_trade_pct": req.risk_per_trade_pct,
        }
        if req.min_volume is not None:
            strategy["min_volume"] = req.min_volume

        if supabase is None:
            result = await run_backtest(req.symbol, strategy, req.date_from, req.date_to, req.initial_capital, req.timeframe)
            return {"status": "DONE", "data": result, "persistence": "none"}

        # Crear registro en Supabase con status PENDING
        record = supabase.table("backtests").insert({
            "name":           req.name,
            "symbol":         req.symbol.upper(),
            "strategy":       json.dumps(strategy),
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
            backtest_id, req.symbol, strategy,
            req.date_from, req.date_to, req.initial_capital, req.timeframe
        )

        return {"backtest_id": backtest_id, "status": "RUNNING"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_and_save_backtest(backtest_id, symbol, strategy, date_from, date_to, capital, timeframe):
    """Tarea de background: corre backtest y actualiza Supabase."""
    if supabase is None:
        return
    try:
        result = await run_backtest(symbol, strategy, date_from, date_to, capital, timeframe)
        update_payload = {
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
            "metrics":          json.dumps({
                "engine_version": result.get("engine_version"),
                "buy_and_hold_return_pct": result.get("buy_and_hold_return_pct"),
                "number_of_trades": result.get("number_of_trades"),
                "average_trade_return": result.get("average_trade_return"),
                "average_win": result.get("average_win"),
                "average_loss": result.get("average_loss"),
                "profit_factor": result.get("profit_factor"),
                "expectancy": result.get("expectancy"),
                "fees_total": result.get("fees_total"),
                "slippage_total": result.get("slippage_total"),
                "assumptions": result.get("assumptions", {}),
            }),
            "status":           "DONE",
            "finished_at":      datetime.now(timezone.utc).isoformat(),
        }
        try:
            supabase.table("backtests").update(update_payload).eq("id", backtest_id).execute()
        except Exception:
            update_payload.pop("metrics", None)
            supabase.table("backtests").update(update_payload).eq("id", backtest_id).execute()
    except Exception as e:
        supabase.table("backtests").update({
            "status": "ERROR", "error_msg": str(e)
        }).eq("id", backtest_id).execute()


@app.get("/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """Obtiene resultado de un backtest por ID."""
    try:
        if supabase is None:
            raise HTTPException(status_code=503, detail="Supabase is required for persisted async backtests")
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
        if supabase is None:
            raise HTTPException(status_code=503, detail="Supabase is required for persisted async backtests")
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
