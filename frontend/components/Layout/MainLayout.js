import React from 'react';
import { Layout, Menu } from 'antd';
import {
  HomeOutlined,
  ProjectOutlined,
  SettingOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/router';
import Link from 'next/link';

const { Header, Content, Sider } = Layout;

const MainLayout = ({ children, selectedKey }) => {
  const router = useRouter();

  const menuItems = [
    {
      key: 'dashboard',
      icon: <HomeOutlined />,
      label: '仪表盘',
      path: '/admin/dashboard',
    },
    {
      key: 'admin-projects',
      icon: <ProjectOutlined />,
      label: '项目管理',
      path: '/admin/projects',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '系统设置',
      path: '/admin/settings',
    },
    {
      key: 'users',
      icon: <UserOutlined />,
      label: '用户管理',
      path: '/admin/users',
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ 
        padding: '0 24px', 
        background: '#fff', 
        borderBottom: '1px solid #f0f0f0',
        display: 'flex',
        alignItems: 'center',
      }}>
        <div style={{ 
          color: '#1890ff', 
          fontSize: '18px', 
          fontWeight: 'bold' 
        }}>
          PropertyWize 管理系统
        </div>
      </Header>
      <Layout>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            style={{ height: '100%', borderRight: 0 }}
          >
            {menuItems.map(item => (
              <Menu.Item key={item.key} icon={item.icon}>
                <Link href={item.path}>
                  {item.label}
                </Link>
              </Menu.Item>
            ))}
          </Menu>
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content style={{
            background: '#fff',
            padding: 24,
            margin: 0,
            minHeight: 280,
          }}>
            {children}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default MainLayout; 