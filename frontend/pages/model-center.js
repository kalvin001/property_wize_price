import { useState } from 'react';
import { 
  Breadcrumb, 
  Typography,
  Tabs,
  Card
} from 'antd';
import {
  BarChartOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import Link from 'next/link';
import Head from 'next/head';
import AppLayout from '../components/Layout/AppLayout';
import ModelEvaluation from '../components/ModelCenter/ModelEvaluation';
import ModelManagement from '../components/ModelCenter/ModelManagement';

const { Title } = Typography;

const ModelCenter = () => {
  const [activeTab, setActiveTab] = useState('evaluation');

  const items = [
    {
      key: 'evaluation',
      label: '模型评估',
      children: <ModelEvaluation />,
      icon: <BarChartOutlined />
    },
    {
      key: 'management',
      label: '模型管理',
      children: <ModelManagement />,
      icon: <SettingOutlined />
    }
  ];

  return (
    <AppLayout>
      <Head>
        <title>模型中心 - PropertyWize</title>
      </Head>
      
      <Breadcrumb style={{ margin: '16px 0' }}>
        <Breadcrumb.Item>
          <Link href="/" legacyBehavior>首页</Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item>模型中心</Breadcrumb.Item>
      </Breadcrumb>
      
      <div style={{ background: '#fff', padding: 24, minHeight: 280 }}>
        <Title level={2}>模型中心</Title>
        <p>欢迎使用 PropertyWize 模型中心，这里集中了所有模型相关的管理和评估功能。</p>
        
        <Card>
          <Tabs 
            activeKey={activeTab} 
            onChange={setActiveTab}
            items={items}
            type="card"
            size="large"
            tabBarGutter={8}
          />
        </Card>
      </div>
    </AppLayout>
  );
};

export default ModelCenter; 