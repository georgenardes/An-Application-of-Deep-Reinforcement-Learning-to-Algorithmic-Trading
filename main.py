# coding=utf-8

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse
import json
import os
from datetime import datetime

from tradingSimulator import TradingSimulator
from tradingPerformance import PerformanceEstimator



###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # Ensure output directories exist
    os.makedirs('Figures', exist_ok=True)

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
    parser.add_argument("-start_date", default='2012-01-01', type=str, help="Start date of the stock market data (format: YYYY-MM-DD)")
    parser.add_argument("-end_date", default='2025-03-31', type=str, help="End date of the stock market data (format: YYYY-MM-DD)")
    parser.add_argument("-split_date", default='2024-01-01', type=str, help="Date separating the training and testing periods (format: YYYY-MM-DD)")
    args = parser.parse_args()
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock

    # Training and testing of the trading strategy specified for the stock (market) specified
    _, _, testingEnv = simulator.simulateNewStrategy(strategy, stock, saveStrategy=False, startingDate=args.start_date, endingDate=args.end_date, splitingDate=args.split_date)

    # Compute and save performance metrics to JSON
    analyser = PerformanceEstimator(testingEnv.data)
    analyser.computePerformance()
    metrics = {
        "strategy": strategy,
        "stock": stock,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "split_date": args.split_date,
        "timestamp": datetime.now().isoformat(),
        "PnL": analyser.PnL,
        "annualized_return_pct": analyser.annualizedReturn,
        "annualized_volatility_pct": analyser.annualizedVolatily,
        "sharpe_ratio": analyser.sharpeRatio,
        "sortino_ratio": analyser.sortinoRatio,
        "max_drawdown_pct": analyser.maxDD,
        "max_drawdown_duration_days": analyser.maxDDD,
        "profitability_pct": analyser.profitability,
        "avg_profit_loss_ratio": analyser.averageProfitLossRatio if analyser.averageProfitLossRatio != float('Inf') else None,
        "skewness": analyser.skewness,
    }
    os.makedirs('Results', exist_ok=True)
    metrics_path = os.path.join('Results', f"{strategy}_{stock}_{args.split_date}_{args.end_date}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4, default=lambda x: float(x) if hasattr(x, '__float__') else int(x))
    print(f"Metrics saved to {metrics_path}")

    """
    simulator.displayTestbench()
    simulator.analyseTimeSeries(stock)
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=False)
    simulator.simulateExistingStrategy(strategy, stock)
    simulator.evaluateStrategy(strategy, saveStrategy=False)
    simulator.evaluateStock(stock)
    """
