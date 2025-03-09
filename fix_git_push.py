#!/usr/bin/env python
"""
修复Git推送问题
"""
import subprocess
import time

def run_command(command):
    """运行命令并返回结果"""
    print(f"\n执行: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"错误: {result.stderr}")
        return bool(result.returncode == 0)
    except Exception as e:
        print(f"命令执行出错: {str(e)}")
        return False

def fix_git_push():
    """修复Git推送问题"""
    print("=" * 50)
    print("Git推送修复工具")
    print("=" * 50)
    
    # 1. 检查远程仓库
    print("\n步骤1: 检查远程仓库配置...")
    run_command("git remote -v")
    
    # 2. 重命名远程仓库
    print("\n步骤2: 重命名当前的远程仓库为'old-origin'...")
    run_command("git remote rename origin old-origin")
    
    # 3. 创建新的本地分支
    repo_url = input("\n请输入您的GitHub仓库URL (格式: git@github.com:username/repo.git): ")
    if not repo_url:
        print("未提供仓库URL，使用默认值")
        repo_url = "git@github.com:kalvin001/property_wize_price.git"
    
    # 4. 完全清除远程关联
    print("\n步骤3: 移除所有远程仓库关联...")
    run_command("git remote remove old-origin")
    
    # 5. 添加新的远程仓库
    print("\n步骤4: 添加新的远程仓库...")
    run_command(f"git remote add origin {repo_url}")
    
    # 6. 确保当前分支是main
    print("\n步骤5: 确保当前分支是main...")
    run_command("git branch -M main")
    
    # 7. 强制推送
    print("\n步骤6: 强制推送到远程仓库...")
    push_result = run_command("git push -f origin main")
    
    if push_result:
        print("\n✅ 成功! 您的代码已成功推送到GitHub仓库。")
    else:
        # 提供额外的解决方案
        print("\n❌ 推送失败。尝试以下额外步骤:")
        print("\n选项1: 删除远程仓库上的所有内容")
        print("请登录到GitHub，进入仓库设置中的'Danger Zone'，选择'Delete this repository'。")
        print("然后重新创建一个同名仓库，再运行此脚本。")
        
        print("\n选项2: 使用以下指令手动强制推送:")
        print(f"git remote remove origin")
        print(f"git remote add origin {repo_url}")
        print("git push -f origin main")

if __name__ == "__main__":
    fix_git_push() 