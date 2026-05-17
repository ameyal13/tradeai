# backend/agents/trading_agent.py
"""
Agente de Trading Copilot con LangGraph.
Usa múltiples proveedores gratuitos (Groq, Google, OpenRouter) con fallback automático.
Tools: análisis de mercado, noticias, señales, backtesting.
"""
import os
import json
import asyncio
from typing import TypedDict, Annotated, Sequence
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

from tools.market_tool import get_market_analysis, get_top_cryptos, get_current_price
from tools.news_tool import get_news


# ── LLM Factory — intercambia proveedores sin cambiar código ──────────────

def get_llm(provider: str = "groq", model: str = None):
    """
    Retorna un LLM con interfaz OpenAI-compatible.
    Groq y OpenRouter usan el mismo SDK de OpenAI con base_url diferente.
    """
    if provider == "groq":
        return ChatOpenAI(
            model=model or os.getenv("FAST_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1,
            max_tokens=2048,
        )
    elif provider == "google":
        return ChatOpenAI(
            model=model or os.getenv("ANALYSIS_MODEL", "gemini-2.0-flash"),
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.1,
            max_tokens=2048,
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            model=model or os.getenv("REASONING_MODEL", "deepseek/deepseek-r1:free"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=2048,
            default_headers={"HTTP-Referer": "https://trading-copilot.app"},
        )
    # fallback
    return get_llm("groq")


# ── LangGraph Tools ────────────────────────────────────────────────────────

@tool
async def analyze_market(symbol: str, interval: str = "1h") -> str:
    """
    Analiza el mercado para un activo cripto específico.
    Retorna precio actual, RSI, MACD, Bollinger Bands y señales técnicas.
    Uso: cuando el usuario pregunta sobre el estado actual de un activo.
    symbol: símbolo del activo (BTC, ETH, SOL, etc.)
    interval: 1m, 5m, 15m, 1h, 4h, 1d
    """
    result = await get_market_analysis(symbol, interval)
    return json.dumps(result, indent=2)


@tool
async def get_crypto_news(symbol: str = None, limit: int = 10) -> str:
    """
    Obtiene noticias recientes del mercado cripto.
    Uso: cuando el usuario pregunta sobre noticias o quiere contexto de mercado.
    symbol: activo específico (BTC, ETH) o None para todas las noticias
    """
    news = await get_news(symbol=symbol, limit=limit)
    return json.dumps(news, indent=2)


@tool
async def get_market_overview() -> str:
    """
    Retorna overview del mercado: top 10 criptos por market cap.
    Uso: cuando el usuario pide un resumen general del mercado.
    """
    top = await get_top_cryptos(limit=10)
    return json.dumps(top, indent=2)


@tool
async def generate_signal(symbol: str, interval: str = "1h") -> str:
    """
    Genera una señal de trading (BUY/SELL/HOLD) basada en análisis técnico completo.
    Uso: cuando el usuario pide una recomendación concreta de inversión.
    IMPORTANTE: Siempre incluye disclaimer de que no es asesoramiento financiero.
    symbol: activo a analizar (BTC, ETH, SOL, etc.)
    """
    market = await get_market_analysis(symbol, interval)
    news   = await get_news(symbol=symbol, limit=5)

    context = {
        "market_data": market,
        "recent_news": news,
        "timestamp":   datetime.utcnow().isoformat(),
    }
    return json.dumps(context, indent=2)


TOOLS = [analyze_market, get_crypto_news, get_market_overview, generate_signal]


# ── LangGraph State & Graph ────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


SYSTEM_PROMPT = """Eres un analista de trading cripto experto. Ayudas al usuario a tomar mejores decisiones de inversión combinando análisis técnico, noticias y contexto de mercado.

REGLAS IMPORTANTES:
1. Siempre usa las herramientas disponibles para obtener datos REALES y actualizados antes de responder
2. Explica tus razonamientos de forma clara y educativa
3. Cuando des señales, SIEMPRE incluye nivel de confianza (0-100%) y el razonamiento
4. SIEMPRE incluye este disclaimer: "⚠️ Esto no es asesoramiento financiero. Invierte solo lo que puedas perder."
5. Cuando el RSI < 30, menciona zona de sobreventa. Cuando > 70, zona de sobrecompra.
6. Explica qué significa cada indicador para que el usuario aprenda
7. Si las noticias son negativas, mencionalo como factor de riesgo adicional

FORMATO DE SEÑALES:
- Señal: BUY / SELL / HOLD
- Confianza: X%
- Precio actual: $X
- Stop Loss sugerido: $X (-X%)
- Take Profit sugerido: $X (+X%)
- Razonamiento: [análisis técnico + contexto de noticias]"""


def build_agent(provider: str = "groq"):
    """Construye el grafo LangGraph con el proveedor especificado."""
    llm = get_llm(provider)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Análisis de señal standalone ──────────────────────────────────────────

async def generate_trading_signal(symbol: str, interval: str = "1h", provider: str = "groq") -> dict:
    """
    Genera una señal estructurada para guardar en Supabase.
    Usado por el endpoint /signals/generate.
    """
    market = await get_market_analysis(symbol, interval)
    news   = await get_news(symbol=symbol, limit=5)

    if "error" in market:
        return {"error": market["error"]}

    indicators = market.get("indicators", {})
    analysis   = market.get("analysis", {})

    # Lógica de señal basada en indicadores (reglas deterministas primero)
    score = 0
    reasons = []

    rsi = indicators.get("rsi", 50)
    if rsi < 30:
        score += 2; reasons.append(f"RSI={rsi:.1f} en zona de sobreventa (señal alcista)")
    elif rsi > 70:
        score -= 2; reasons.append(f"RSI={rsi:.1f} en zona de sobrecompra (señal bajista)")
    else:
        reasons.append(f"RSI={rsi:.1f} neutral")

    if analysis.get("macd_signal") == "BULLISH":
        score += 1; reasons.append("MACD positivo (momentum alcista)")
    else:
        score -= 1; reasons.append("MACD negativo (momentum bajista)")

    bb_pos = analysis.get("bb_position", "")
    if bb_pos == "BELOW_LOWER":
        score += 2; reasons.append("Precio bajo banda inferior de Bollinger (posible rebote)")
    elif bb_pos == "ABOVE_UPPER":
        score -= 2; reasons.append("Precio sobre banda superior de Bollinger (posible corrección)")

    news_sentiment = sum(1 for n in news if "bull" in n.get("title", "").lower()) - \
                     sum(1 for n in news if "bear" in n.get("title", "").lower() or "crash" in n.get("title", "").lower())
    if news_sentiment > 0:
        score += 1; reasons.append("Sentimiento de noticias positivo")
    elif news_sentiment < 0:
        score -= 1; reasons.append("Sentimiento de noticias negativo")

    # Señal final
    if score >= 2:
        signal_type = "BUY"
        confidence  = min(50 + score * 10, 90)
    elif score <= -2:
        signal_type = "SELL"
        confidence  = min(50 + abs(score) * 10, 90)
    else:
        signal_type = "HOLD"
        confidence  = 40 + abs(score) * 5

    price = indicators.get("price", 0)
    return {
        "symbol":          symbol.upper(),
        "signal_type":     signal_type,
        "confidence":      confidence,
        "price_at_signal": price,
        "entry_price":     price,
        "stop_loss":       round(price * 0.95, 6),   # 5% stop loss por defecto
        "take_profit":     round(price * 1.10, 6),   # 10% take profit por defecto
        "reasoning":       " | ".join(reasons),
        "indicators":      indicators,
        "news_context":    [{"title": n["title"], "source": n["source"]} for n in news[:3]],
        "model_used":      f"rule-based+{provider}",
    }