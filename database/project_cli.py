import argparse
import os
import sys
import json
from tabulate import tabulate
from .project_manager import project_manager

def main():
    """项目管理命令行工具主函数"""
    parser = argparse.ArgumentParser(description='PropertyWize项目管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出项目命令
    list_parser = subparsers.add_parser('list', help='列出所有项目')
    
    # 创建项目命令
    create_parser = subparsers.add_parser('create', help='创建新项目')
    create_parser.add_argument('name', help='项目名称')
    create_parser.add_argument('-d', '--description', help='项目描述')
    
    # 扫描项目命令
    scan_parser = subparsers.add_parser('scan', help='扫描项目结构')
    scan_parser.add_argument('-n', '--name', help='项目名称')
    scan_parser.add_argument('-i', '--id', type=int, help='项目ID')
    
    # 删除项目命令
    delete_parser = subparsers.add_parser('delete', help='删除项目')
    delete_parser.add_argument('id', type=int, help='项目ID')
    
    # 查看项目详情命令
    show_parser = subparsers.add_parser('show', help='查看项目详情')
    show_parser.add_argument('id', type=int, help='项目ID')
    show_parser.add_argument('-f', '--files', action='store_true', help='显示项目文件列表')
    
    # 导出项目命令
    export_parser = subparsers.add_parser('export', help='导出项目')
    export_parser.add_argument('id', type=int, help='项目ID')
    export_parser.add_argument('-o', '--output', default='project_export.zip', help='输出文件路径')
    
    # 导入项目命令
    import_parser = subparsers.add_parser('import', help='导入项目')
    import_parser.add_argument('file', help='项目ZIP文件路径')
    
    args = parser.parse_args()
    
    # 处理命令
    if args.command == 'list':
        list_projects()
    elif args.command == 'create':
        create_project(args.name, args.description)
    elif args.command == 'scan':
        scan_project(args.name, args.id)
    elif args.command == 'delete':
        delete_project(args.id)
    elif args.command == 'show':
        show_project(args.id, args.files)
    elif args.command == 'export':
        export_project(args.id, args.output)
    elif args.command == 'import':
        import_project(args.file)
    else:
        parser.print_help()

def list_projects():
    """列出所有项目"""
    projects = project_manager.get_all_projects()
    
    if not projects:
        print("没有找到任何项目")
        return
    
    table_data = []
    for project in projects:
        created_at = project.created_at.strftime("%Y-%m-%d %H:%M:%S")
        updated_at = project.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        table_data.append([
            project.id,
            project.name,
            project.description[:30] + "..." if project.description and len(project.description) > 30 else project.description,
            created_at,
            updated_at
        ])
    
    print(tabulate(
        table_data,
        headers=["ID", "项目名称", "描述", "创建时间", "更新时间"],
        tablefmt="grid"
    ))

def create_project(name, description=None):
    """创建新项目"""
    project = project_manager.create_project(name, description)
    print(f"项目创建成功：ID={project.id}, 名称={project.name}")

def scan_project(name=None, project_id=None):
    """扫描项目结构"""
    try:
        if not name and not project_id:
            print("错误：必须提供项目名称或项目ID")
            return
            
        project = project_manager.scan_project_structure(name, project_id)
        print(f"已完成项目扫描：ID={project.id}, 名称={project.name}")
        
        # 显示扫描结果摘要
        if project.config and 'files' in project.config:
            file_count = len(project.config['files'])
            print(f"扫描的文件数：{file_count}")
            
        if project.config and 'structure' in project.config:
            structure = project.config['structure']
            print(f"前端框架：{structure.get('frontend_framework', 'unknown')}")
            print(f"后端框架：{structure.get('backend_framework', 'unknown')}")
    
    except Exception as e:
        print(f"扫描项目时出错：{e}")

def delete_project(project_id):
    """删除项目"""
    try:
        confirm = input(f"确定要删除ID为{project_id}的项目吗？此操作不可撤销！(y/n): ")
        if confirm.lower() != 'y':
            print("已取消删除操作")
            return
            
        success = project_manager.delete_project(project_id)
        if success:
            print(f"项目ID={project_id}已成功删除")
        else:
            print(f"未找到ID为{project_id}的项目")
    
    except Exception as e:
        print(f"删除项目时出错：{e}")

def show_project(project_id, show_files=False):
    """查看项目详情"""
    try:
        project = project_manager.get_project(project_id)
        if not project:
            print(f"未找到ID为{project_id}的项目")
            return
            
        print(f"项目ID: {project.id}")
        print(f"项目名称: {project.name}")
        print(f"项目描述: {project.description}")
        print(f"创建时间: {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"更新时间: {project.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if project.config:
            if 'structure' in project.config:
                print("\n项目结构:")
                structure = project.config['structure']
                for key, value in structure.items():
                    print(f"  {key}: {value}")
            
            if show_files and 'files' in project.config and project.config['files']:
                print("\n项目文件:")
                file_data = []
                for file_info in project.config['files']:
                    file_data.append([
                        file_info['path'],
                        file_info['type'],
                        f"{file_info['size']} 字节",
                        file_info.get('last_modified', 'N/A')
                    ])
                
                print(tabulate(
                    file_data,
                    headers=["文件路径", "类型", "大小", "最后修改时间"],
                    tablefmt="grid"
                ))
    
    except Exception as e:
        print(f"查看项目详情时出错：{e}")

def export_project(project_id, output_path):
    """导出项目"""
    try:
        output_file = project_manager.export_project(project_id, output_path)
        print(f"项目已成功导出到：{output_file}")
    
    except Exception as e:
        print(f"导出项目时出错：{e}")

def import_project(file_path):
    """导入项目"""
    try:
        project = project_manager.import_project(file_path)
        print(f"项目已成功导入：ID={project.id}, 名称={project.name}")
    
    except Exception as e:
        print(f"导入项目时出错：{e}")

if __name__ == "__main__":
    main() 