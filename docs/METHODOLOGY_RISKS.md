# Methodology Risks

Trading Copilot is research software. It does not guarantee profitability, and outputs must not be treated as financial advice.

## Key Risks

1. **Overfitting**  
   A strategy can look strong on historical data because parameters were fitted to noise.

2. **Look-ahead bias**  
   Signals must only use information available at the time of the signal. Future candles are allowed only during outcome evaluation.

3. **Survivorship bias**  
   Testing only currently popular symbols can hide failures from delisted or illiquid assets.

4. **Incomplete or incorrect candles**  
   Public APIs can return missing, delayed, revised, or rate-limited data.

5. **Optimizing only win rate**  
   A high win rate can still lose money if losses are much larger than wins.

6. **Transaction costs**  
   Fees, slippage, and spread can turn apparently profitable strategies negative.

7. **Regime change**  
   Crypto market behavior can change quickly. Historical performance can decay.

## Required Validation

Use walk-forward validation:

- tune on a train window;
- validate on a later window;
- compare candidate parameters against a baseline;
- mark a candidate as better only if it beats the baseline on validation.

Before trusting any strategy, review at minimum:

- `profit_factor`
- `max_drawdown`
- `expectancy`
- `average_return_pct`
- `evaluated_predictions`
- fees, slippage, and spread assumptions
- performance versus baseline

## Paper Trading First

No real-money trading should be enabled until:

- backtests are robust;
- replay results are stable;
- paper trading confirms behavior live;
- risk controls exist;
- credentials are properly scoped;
- the user explicitly requests real execution.

## Why No Neural Networks Yet

Neural networks are intentionally excluded for now because the system first needs:

- reliable labels;
- walk-forward validation;
- robust replay;
- transparent baselines;
- enough evaluated predictions to compare against simple methods.

Simple, auditable methods are easier to falsify and improve at this stage.
