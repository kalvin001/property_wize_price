#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, List, Tuple
import pandas as pd


def count_file_stats(file_path: str) -> Tuple[int, int, int]:
    """
    统计单个文件的行数、字符数和非空行数
    
    Args:
        file_path: 文件路径
        
    Returns:
        (行数, 字符数, 非空行数)的元组
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            total_lines = len(lines)
            total_chars = len(content)
            non_empty_lines = sum(1 for line in lines if line.strip())
            return total_lines, total_chars, non_empty_lines
    except UnicodeDecodeError:
        # 如果不是文本文件，则跳过
        return 0, 0, 0
    except Exception as e:
        print(f"无法读取文件 {file_path}: {str(e)}")
        return 0, 0, 0


def get_file_extension(file_path: str) -> str:
    """获取文件扩展名"""
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def scan_directory(directory: str, exclude_dirs: List[str] = None, include_exts: List[str] = None) -> Dict:
    """
    扫描目录，统计代码行数和字符数
    
    Args:
        directory: 要扫描的目录
        exclude_dirs: 要排除的目录列表
        include_exts: 要包含的文件扩展名列表
        
    Returns:
        包含统计信息的字典
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', '.idea', 'venv', 'node_modules']
    
    if include_exts is None:
        include_exts = ['.py', '.js', '.ts', '.html', '.css', '.json', '.md', '.txt']
    
    stats = {
        'files': [],
        'total_lines': 0,
        'total_chars': 0,
        'total_non_empty_lines': 0,
        'extension_stats': {}
    }
    
    for root, dirs, files in os.walk(directory):
        # 排除指定目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, directory)
            file_ext = get_file_extension(file_path)
            
            # 如果指定了包含的扩展名，则只统计这些扩展名的文件
            if include_exts and file_ext not in include_exts:
                continue
            
            lines, chars, non_empty_lines = count_file_stats(file_path)
            
            if lines > 0:  # 只统计能成功读取的文件
                stats['files'].append({
                    'path': rel_path,
                    'lines': lines,
                    'chars': chars,
                    'non_empty_lines': non_empty_lines,
                    'extension': file_ext
                })
                
                stats['total_lines'] += lines
                stats['total_chars'] += chars
                stats['total_non_empty_lines'] += non_empty_lines
                
                # 按扩展名统计
                if file_ext not in stats['extension_stats']:
                    stats['extension_stats'][file_ext] = {
                        'files': 0,
                        'lines': 0,
                        'chars': 0,
                        'non_empty_lines': 0
                    }
                
                stats['extension_stats'][file_ext]['files'] += 1
                stats['extension_stats'][file_ext]['lines'] += lines
                stats['extension_stats'][file_ext]['chars'] += chars
                stats['extension_stats'][file_ext]['non_empty_lines'] += non_empty_lines
    
    return stats


def generate_report(stats: Dict, output_format: str = 'text') -> str:
    """
    生成统计报告
    
    Args:
        stats: 统计信息字典
        output_format: 输出格式 ('text'|'markdown'|'csv')
        
    Returns:
        格式化的报告字符串
    """
    if output_format == 'text':
        report = "项目代码统计\n"
        report += "=" * 50 + "\n\n"
        
        report += f"总文件数: {len(stats['files'])}\n"
        report += f"总行数: {stats['total_lines']}\n"
        report += f"总字符数: {stats['total_chars']}\n"
        report += f"总非空行数: {stats['total_non_empty_lines']}\n\n"
        
        report += "按文件类型统计:\n"
        report += "-" * 50 + "\n"
        report += f"{'扩展名':<10}{'文件数':<10}{'行数':<10}{'字符数':<15}{'非空行数':<10}{'平均行数':<10}\n"
        
        for ext, ext_stats in sorted(stats['extension_stats'].items(), 
                                     key=lambda x: x[1]['lines'], 
                                     reverse=True):
            avg_lines = ext_stats['lines'] / ext_stats['files'] if ext_stats['files'] > 0 else 0
            report += f"{ext:<10}{ext_stats['files']:<10}{ext_stats['lines']:<10}"
            report += f"{ext_stats['chars']:<15}{ext_stats['non_empty_lines']:<10}{avg_lines:.1f}\n"
        
        report += "\n最大的10个文件:\n"
        report += "-" * 50 + "\n"
        report += f"{'文件':<50}{'行数':<10}{'字符数':<15}{'非空行数':<10}\n"
        
        for file in sorted(stats['files'], key=lambda x: x['lines'], reverse=True)[:10]:
            report += f"{file['path'][:50]:<50}{file['lines']:<10}{file['chars']:<15}{file['non_empty_lines']:<10}\n"
    
    elif output_format == 'markdown':
        report = "# 项目代码统计\n\n"
        
        report += f"- 总文件数: {len(stats['files'])}\n"
        report += f"- 总行数: {stats['total_lines']}\n"
        report += f"- 总字符数: {stats['total_chars']}\n"
        report += f"- 总非空行数: {stats['total_non_empty_lines']}\n\n"
        
        report += "## 按文件类型统计\n\n"
        report += "| 扩展名 | 文件数 | 行数 | 字符数 | 非空行数 | 平均行数 |\n"
        report += "| ------ | ------ | ---- | ------ | -------- | -------- |\n"
        
        for ext, ext_stats in sorted(stats['extension_stats'].items(), 
                                     key=lambda x: x[1]['lines'], 
                                     reverse=True):
            avg_lines = ext_stats['lines'] / ext_stats['files'] if ext_stats['files'] > 0 else 0
            report += f"| {ext} | {ext_stats['files']} | {ext_stats['lines']} | "
            report += f"{ext_stats['chars']} | {ext_stats['non_empty_lines']} | {avg_lines:.1f} |\n"
        
        report += "\n## 最大的10个文件\n\n"
        report += "| 文件 | 行数 | 字符数 | 非空行数 |\n"
        report += "| ---- | ---- | ------ | -------- |\n"
        
        for file in sorted(stats['files'], key=lambda x: x['lines'], reverse=True)[:10]:
            report += f"| {file['path']} | {file['lines']} | {file['chars']} | {file['non_empty_lines']} |\n"
    
    elif output_format == 'csv':
        # 生成扩展名统计的DataFrame
        ext_data = []
        for ext, ext_stats in stats['extension_stats'].items():
            avg_lines = ext_stats['lines'] / ext_stats['files'] if ext_stats['files'] > 0 else 0
            ext_data.append({
                'extension': ext,
                'files': ext_stats['files'],
                'lines': ext_stats['lines'],
                'chars': ext_stats['chars'],
                'non_empty_lines': ext_stats['non_empty_lines'],
                'avg_lines': avg_lines
            })
        
        ext_df = pd.DataFrame(ext_data)
        ext_csv = ext_df.to_csv(index=False)
        
        # 生成文件统计的DataFrame
        files_df = pd.DataFrame(stats['files'])
        files_csv = files_df.to_csv(index=False)
        
        report = "Extension Stats:\n" + ext_csv + "\n\nFile Stats:\n" + files_csv
    
    else:
        report = "不支持的输出格式"
    
    return report


def main():
    parser = argparse.ArgumentParser(description="统计项目代码行数和字符数")
    parser.add_argument('directory', nargs='?', default='.', help='要扫描的目录 (默认为当前目录)')
    parser.add_argument('--exclude', '-e', nargs='+', help='要排除的目录列表')
    parser.add_argument('--include', '-i', nargs='+', help='要包含的文件扩展名列表')
    parser.add_argument('--format', '-f', choices=['text', 'markdown', 'csv'], default='text', 
                        help='输出格式 (text|markdown|csv)')
    parser.add_argument('--output', '-o', help='输出文件路径 (默认为终端输出)')
    
    args = parser.parse_args()
    
    stats = scan_directory(args.directory, args.exclude, args.include)
    report = generate_report(stats, args.format)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"统计报告已保存到: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main() 