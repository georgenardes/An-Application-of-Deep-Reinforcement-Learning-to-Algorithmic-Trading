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
import sys
import time
from datetime import datetime

import pandas as pd

import tradingSimulator as simulatorModule
from tradingSimulator import TradingSimulator
from tradingPerformance import PerformanceEstimator



###############################################################################
##################################### MAIN ####################################
###############################################################################

CHINESE_STOCKS = ['000063', '002008', '002352', '300015', '600030', '600036', '600900', '601899']
BRAZILIAN_STOCKS = ['PETR4', 'CMIG4', 'VALE3', 'TOTS3', 'ITUB4', 'MOTV3', 'RADL3']


def prepare_chinese_csv(stock, start_date, split_date, end_date):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(project_dir, '..'))
    source_path = os.path.join(workspace_dir, 'Data', 'chinese', f'{stock}.csv')

    if not os.path.isfile(source_path):
        return

    data = pd.read_csv(source_path, parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    data.index.names = ['Timestamp']
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    train_data = data.loc[pd.Timestamp(start_date):pd.Timestamp(split_date)]
    test_data = data.loc[pd.Timestamp(split_date):pd.Timestamp(end_date)]

    os.makedirs('Data', exist_ok=True)
    train_path = os.path.join('Data', f'{stock}_{start_date}_{split_date}.csv')
    test_path = os.path.join('Data', f'{stock}_{split_date}_{end_date}.csv')
    train_data.to_csv(train_path)
    test_data.to_csv(test_path)


if(__name__ == '__main__'):

    # Ensure output directories exist
    os.makedirs('Figures', exist_ok=True)

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
    parser.add_argument("-market", default='single', type=str, choices=['single', 'brazilian', 'chinese'], help="Run a single stock or a full market basket")
    parser.add_argument("-start_date", default='2012-01-01', type=str, help="Start date of the stock market data (format: YYYY-MM-DD)")
    parser.add_argument("-end_date", default='2025-03-31', type=str, help="End date of the stock market data (format: YYYY-MM-DD)")
    parser.add_argument("-split_date", default='2024-01-01', type=str, help="Date separating the training and testing periods (format: YYYY-MM-DD)")
    parser.add_argument("-bounds_min", default=1, type=int, help="Lower bound for classical strategy grid search")
    parser.add_argument("-bounds_max", default=30, type=int, help="Upper bound for classical strategy grid search")
    parser.add_argument("-step", default=1, type=int, help="Step size for classical strategy grid search")
    args = parser.parse_args()
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    if args.market == 'brazilian':
        stocks = BRAZILIAN_STOCKS
    elif args.market == 'chinese':
        stocks = CHINESE_STOCKS
    else:
        stocks = [args.stock]

    for chinese_stock in CHINESE_STOCKS:
        simulatorModule.companies[chinese_stock] = chinese_stock

    os.makedirs('Results', exist_ok=True)

    for stock in stocks:
        print(f"\n{'='*60}", flush=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Stock: {stock}  |  Strategy: {strategy}", flush=True)
        print(f"{'='*60}", flush=True)

        if stock in CHINESE_STOCKS:
            print(f"  [1/4] Preparing Chinese CSV data...", flush=True)
            prepare_chinese_csv(stock, args.start_date, args.split_date, args.end_date)
            print(f"  [1/4] Done.", flush=True)
        else:
            print(f"  [1/4] Using Brazilian/built-in data.", flush=True)

        print(f"  [2/4] Running training+testing phase...", flush=True)
        t0 = time.time()
        _, _, testingEnv = simulator.simulateNewStrategy(
            strategy,
            stock,
            saveStrategy=False,
            startingDate=args.start_date,
            endingDate=args.end_date,
            splitingDate=args.split_date,
            bounds=[args.bounds_min, args.bounds_max],
            step=args.step,
            verbose=True,
            plotTraining=False,
            rendering=False,
            showPerformance=False,
        )
        print(f"  [2/4] Done in {time.time()-t0:.1f}s.", flush=True)

        # Compute and save performance metrics to JSON
        print(f"  [3/4] Computing performance metrics...", flush=True)
        analyser = PerformanceEstimator(testingEnv.data)
        analyser.computePerformance()
        print(f"  [3/4] Done. Return={analyser.annualizedReturn:.2f}%  Sharpe={analyser.sharpeRatio:.3f}  MaxDD={analyser.maxDD:.2f}%", flush=True)
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
        metrics_path = os.path.join('Results', f"{strategy}_{stock}_{args.split_date}_{args.end_date}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=lambda x: float(x) if hasattr(x, '__float__') else int(x))
        print(f"  [4/4] Metrics saved to {metrics_path}", flush=True)

    """
    simulator.displayTestbench()
    simulator.analyseTimeSeries(stock)
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=False)
    simulator.simulateExistingStrategy(strategy, stock)
    simulator.evaluateStrategy(strategy, saveStrategy=False)
    simulator.evaluateStock(stock)
    """
