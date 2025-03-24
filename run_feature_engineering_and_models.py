import os
import subprocess
import argparse
import time
import json
from pathlib import Path

def run_command(command, verbose=True):
    """运行命令并返回状态码和输出"""
    if verbose:
        print(f"执行命令: {command}")
    
    # 设置环境变量，确保使用UTF-8编码
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # 使用subprocess.PIPE捕获输出
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        env=env,
        # 明确指定编码
        text=True,
        encoding="utf-8",
        errors="replace"  # 替换无法解码的字符
    )
    
    # 读取输出
    stdout, stderr = process.communicate()
    
    # 获取返回码
    return_code = process.returncode
    
    if return_code != 0:
        print(f"命令执行失败，返回码: {return_code}")
        print(f"错误输出: {stderr}")
    elif verbose:
        print(stdout)
    
    return return_code, stdout, stderr

def run_feature_engineering(data_file, output_file, skip_feature_engineering=False):
    """运行特征工程"""
    if skip_feature_engineering:
        print("跳过特征工程步骤")
        return 0
    
    print("\n==================================================")
    print("执行步骤: 特征工程")
    command = f"python feature_engineering.py --data_file {data_file} --output_file {output_file}"
    print(f"命令: {command}")
    print("==================================================")
    
    start_time = time.time()
    return_code, _, _ = run_command(command, verbose=False)
    elapsed_time = time.time() - start_time
    
    if return_code == 0:
        print(f"✓ 步骤完成，耗时: {elapsed_time:.2f}秒")
    else:
        print(f"✗ 步骤失败，耗时: {elapsed_time:.2f}秒")
    
    return return_code

def train_enhanced_model(model_types, data_file, output_dir, feature_selection=False, optimize_hyperparams=False):
    """训练增强模型"""
    print("\n==================================================")
    print("执行步骤: 训练增强模型")
    
    # 构建命令
    command = f"python train_models_enhanced.py --model_types {' '.join(model_types)} --data_file {data_file} --output_dir {output_dir}"
    
    if feature_selection:
        command += " --feature_selection"
        print("添加特征选择以提高模型性能")
    else:
        print("警告: 不添加特征选择以保留所有特征以获得最佳性能")
    
    if optimize_hyperparams:
        command += " --optimize_hyperparams"
        print("启用超参数优化以提高模型性能")
    
    print(f"命令: {command}")
    print("==================================================")
    
    start_time = time.time()
    return_code, _, _ = run_command(command, verbose=False)
    elapsed_time = time.time() - start_time
    
    if return_code == 0:
        print(f"✓ 步骤完成，耗时: {elapsed_time:.2f}秒")
    else:
        print(f"✗ 步骤失败，耗时: {elapsed_time:.2f}秒")
    
    return return_code

def compare_models(baseline_dir, enhanced_dir):
    """比较基线模型和增强模型的性能"""
    print("\n比较基线模型和增强模型的性能...")
    
    # 读取基线模型和增强模型的性能指标
    baseline_metrics_path = os.path.join(baseline_dir, "xgboost_metrics.json")
    enhanced_metrics_path = os.path.join(enhanced_dir, "xgboost_metrics.json")
    
    if not os.path.exists(baseline_metrics_path) or not os.path.exists(enhanced_metrics_path):
        print("无法比较模型，缺少性能指标文件")
        return
    
    with open(baseline_metrics_path, 'r', encoding='utf-8') as f:
        baseline_metrics = json.load(f)
    
    with open(enhanced_metrics_path, 'r', encoding='utf-8') as f:
        enhanced_metrics = json.load(f)
    
    # 比较关键指标
    print("\n模型性能比较:")
    print(f"{'指标':<25} {'增强模型':<18} {'基线模型':<18} {'改进百分比':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ("RMSE", "rmse", False),
        ("MAE", "mae", False),
        ("平均百分比误差", "mean_percentage_error", False),
        ("中位百分比误差", "median_percentage_error", False)
    ]
    
    for metric_name, metric_key, higher_better in metrics_to_compare:
        baseline_value = baseline_metrics.get(metric_key, 0)
        enhanced_value = enhanced_metrics.get(metric_key, 0)
        
        if higher_better:
            improvement = ((enhanced_value - baseline_value) / baseline_value) * 100
        else:
            improvement = ((baseline_value - enhanced_value) / baseline_value) * 100
        
        print(f"{metric_name:<25} {enhanced_value:<15.2f} % {baseline_value:<15.2f} % {improvement:<15.2f} %")
    
    # 比较误差分布
    print("\n误差范围分布比较:")
    print(f"{'误差范围':<15} {'增强模型':<18} {'基线模型':<18} {'改进百分比':<15}")
    print("-" * 60)
    
    baseline_ranges = baseline_metrics.get("error_distribution", {}).get("error_ranges", {})
    enhanced_ranges = enhanced_metrics.get("error_distribution", {}).get("error_ranges", {})
    
    for range_name in ["<5%", "5-10%", "10-15%", "15-20%", ">20%"]:
        baseline_value = baseline_ranges.get(range_name, 0) * 100
        enhanced_value = enhanced_ranges.get(range_name, 0) * 100
        
        if range_name in ["<5%", "5-10%"]:  # 这些范围内，比例越高越好
            improvement = ((enhanced_value - baseline_value) / baseline_value) * 100
        else:  # 这些范围内，比例越低越好
            improvement = ((baseline_value - enhanced_value) / baseline_value) * 100
        
        print(f"{range_name:<15} {enhanced_value:<15.2f} % {baseline_value:<15.2f} % {improvement:<15.2f} %")
    
    # 显示增强模型的前10个重要特征
    print("\n增强模型前10个重要特征:")
    feature_importance = enhanced_metrics.get("feature_importance", [])
    for i, feature in enumerate(feature_importance[:10]):
        print(f"{i+1}. {feature['feature']} (重要性: {feature['importance']:.2e})")

def main():
    parser = argparse.ArgumentParser(description='运行完整房价预测流程')
    parser.add_argument('--data_file', type=str, default='resources/house_samples.csv',
                       help='原始数据文件路径')
    parser.add_argument('--engineered_file', type=str, default='resources/house_samples_engineered_original.csv',
                       help='特征工程后的数据文件路径')
    parser.add_argument('--skip_feature_engineering', action='store_true',
                       help='是否跳过特征工程步骤')
    parser.add_argument('--model_types', nargs='+', default=['xgboost'],
                       help='要训练的模型类型列表')
    parser.add_argument('--feature_selection', action='store_true',
                       help='是否使用特征选择')
    parser.add_argument('--optimize_hyperparams', action='store_true',
                       help='是否优化超参数')
    parser.add_argument('--baseline_dir', type=str, default='model', 
                       help='基线模型目录')
    parser.add_argument('--enhanced_dir', type=str, default='model_original',
                       help='增强模型目录')
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    print("开始执行完整房价预测流程")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 特征工程
    feature_engineering_status = run_feature_engineering(
        args.data_file, 
        args.engineered_file,
        args.skip_feature_engineering
    )
    
    if feature_engineering_status != 0 and not args.skip_feature_engineering:
        print("特征工程步骤失败，流程终止")
        return
    
    # 步骤2: 训练增强模型
    train_status = train_enhanced_model(
        args.model_types,
        args.engineered_file,
        args.enhanced_dir,
        args.feature_selection,
        args.optimize_hyperparams
    )
    
    if train_status != 0:
        print("模型训练步骤失败，流程终止")
        return
    
    # 步骤3: 比较模型
    compare_models(args.baseline_dir, args.enhanced_dir)
    
    # 计算总耗时
    total_time = time.time() - start_time
    print("\n完整流程执行完毕!")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 