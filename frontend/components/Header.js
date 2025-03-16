import { Layout, Menu, Typography } from 'antd';
import Link from 'next/link';
import { HomeOutlined } from '@ant-design/icons';

const { Header } = Layout;

const MainHeader = ({ selectedKey = '1' }) => {
  return (
    <Header style={{ position: 'fixed', zIndex: 1, width: '100%', background: 'white', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
      <div style={{ float: 'left', marginRight: '30px' }}>
        <Typography.Title level={3} style={{ margin: '8px 0', color: '#1890ff' }}>
          <HomeOutlined /> PropertyWize
        </Typography.Title>
      </div>
      <Menu 
        theme="light" 
        mode="horizontal" 
        defaultSelectedKeys={[selectedKey]}
        style={{ lineHeight: '64px' }}
        items={[
          { key: '1', label: <Link href="/">首页</Link> },
          { key: '2', label: <Link href="/model-evaluation">模型评估</Link> },
          { key: '3', label: <Link href="/property-reports">房产估价报告</Link> },
        ]}
      />
    </Header>
  );
};

export default MainHeader; 