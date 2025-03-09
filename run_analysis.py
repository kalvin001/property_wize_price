#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import time
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def visualize_optimization_results(log_dir=None):
    """可视化最近的参数优化结果"""
    if log_dir is None:
        # 获取最新的优化日志目录
        optimizer_logs = glob.glob('model/optimizer_logs/*')
        if not optimizer_logs:
            print("未找到优化日志")
            return
        log_dir = max(optimizer_logs, key=os.path.getctime)
    
    print(f"分析优化日志: {log_dir}")
    
    # 读取优化日志
    log_file = os.path.join(log_dir, 'optimization_log.json')
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        opt_data = json.load(f)
    
    # 提取基本信息
    method = opt_data.get('method', 'unknown')
    start_time = opt_data.get('start_time', '')
    end_time = opt_data.get('end_time', '')
    total_seconds = opt_data.get('total_seconds', 0)
    best_score = opt_data.get('best_score', 0)
    best_params = opt_data.get('best_parameters', {})
    
    print("\n优化摘要:")
    print(f"方法: {method}")
    print(f"开始时间: {start_time}")
    print(f"结束时间: {end_time}")
    print(f"总耗时: {total_seconds:.2f}秒 ({total_seconds/60:.2f}分钟)")
    print(f"最佳得分 (RMSE): {best_score:.4f}")
    print("\n最佳参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # 提取迭代数据
    iterations = opt_data.get('iterations', [])
    if not iterations:
        print("\n未找到迭代数据")
        return
    
    # 创建分析目录
    analysis_dir = os.path.join(log_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 将迭代数据转换为DataFrame
    iter_data = []
    for it in iterations:
        if it.get('score') is None:
            continue
        row = {'iteration': it.get('iteration', 0), 'score': it.get('score', 0), 'time': it.get('elapsed_seconds', 0)}
        params = it.get('parameters', {})
        if isinstance(params, dict):
            for param, value in params.items():
                row[param] = value
        iter_data.append(row)
    
    df = pd.DataFrame(iter_data)
    if df.empty:
        print("\n迭代数据为空")
        return
    
    # 保存分析数据
    df.to_csv(os.path.join(analysis_dir, 'iterations.csv'), index=False)
    
    # 绘制优化进度图
    plt.figure(figsize=(12, 6))
    plt.plot(df['iteration'], df['score'], 'b-', marker='o')
    plt.axhline(y=best_score, color='r', linestyle='--', label=f'最佳得分: {best_score:.4f}')
    plt.title('优化进度 - RMSE随迭代的变化')
    plt.xlabel('迭代')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, 'optimization_progress.png'), dpi=100)
    
    # 绘制参数影响图 (对于贝叶斯优化)
    if method == 'bayes' and len(df.columns) > 3:  # 确保有参数列
        param_cols = [col for col in df.columns if col not in ['iteration', 'score', 'time']]
        n_params = len(param_cols)
        
        if n_params > 0:
            fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
            if n_params == 1:
                axes = [axes]
            
            for i, param in enumerate(param_cols):
                if param in df.columns and df[param].nunique() > 1:
                    axes[i].scatter(df[param], df['score'], alpha=0.7)
                    axes[i].set_title(f'参数 {param} 对RMSE的影响')
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel('RMSE')
                    axes[i].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'parameter_effects.png'), dpi=100)
    
    print(f"\n分析结果保存在: {analysis_dir}")

def main():
    parser = argparse.ArgumentParser(description='运行房产价格预测模型分析')
    parser.add_argument('--preset', type=str, default='default',
                        choices=['default', 'quick', 'thorough', 'grid', 'bayes'],
                        help='预设配置: default(默认), quick(快速), thorough(彻底), grid(网格搜索), bayes(贝叶斯优化)')
    parser.add_argument('--custom_args', type=str, default='',
                        help='传递给analyze_data.py的自定义参数，例如 "--optimization grid --cv_folds 5"')
    parser.add_argument('--visualize', action='store_true',
                        help='分析和可视化最近的优化结果')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='指定要分析的优化日志目录')
    
    args = parser.parse_args()
    
    # 如果只是要可视化，不运行分析
    if args.visualize or args.log_dir:
        visualize_optimization_results(args.log_dir)
        return 0
    
    # 预设配置
    presets = {
        'default': '--optimization bayes --feature_selection --cv_folds 3 --bayes_iterations 20 --min_features 10',
        'quick': '--optimization grid --feature_selection --cv_folds 3 --min_features 15',
        'thorough': '--optimization bayes --feature_selection --cv_folds 5 --bayes_iterations 50 --min_features 5',
        'grid': '--optimization grid --feature_selection --cv_folds 3 --min_features 10',
        'bayes': '--optimization bayes --feature_selection --cv_folds 3 --bayes_iterations 20 --min_features 10'
    }
    
    # 选择要使用的参数
    if args.custom_args:
        cmd_args = args.custom_args
    else:
        cmd_args = presets[args.preset]
    
    # 运行时间
    start_time = time.time()
    
    # 构建并运行命令
    cmd = f"python analyze_data.py {cmd_args}"
    print(f"运行命令: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"\n分析完成! 总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")
        print("\n结果位置:")
        print("- 模型: model/")
        print("- 数据分析结果: frontend/public/data/")
        
        # 自动可视化最新的优化结果
        print("\n正在分析优化结果...")
        visualize_optimization_results()
    except subprocess.CalledProcessError as e:
        print(f"分析过程中发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 