import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Link from 'next/link';
import { 
  Layout, 
  Menu, 
  Typography, 
  Row, 
  Col, 
  Button, 
  Breadcrumb,
  Spin,
  Result,
  Space
} from 'antd';
import { 
  HomeOutlined, 
  ArrowLeftOutlined,
  DollarOutlined,
  ExportOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import PropertyBasicInfo from '../../../components/PropertyBasicInfo';
import PropertyLocation from '../../../components/PropertyLocation';
import PropertyImages from '../../../components/PropertyImages';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

export default function PropertyBasicInfoPage() {
  const router = useRouter();
  const { id } = router.query;
  const [property, setProperty] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 获取房产详情
  useEffect(() => {
    if (!id) return;
    
    const fetchPropertyDetail = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // 这里通常会从API获取数据，为了演示我们使用模拟数据
        const response = await fetch(`/api/properties/${id}`);
        
        if (!response.ok) {
          throw new Error(`获取房产信息失败: ${response.status}`);
        }
        
        const data = await response.json();
        const transformedData = transformPropertyDetail(data);
        setProperty(transformedData);
      } catch (err) {
        console.error('获取房产详情出错:', err);
        setError(err.message || '获取房产信息失败');
      } finally {
        setLoading(false);
      }
    };
    
    fetchPropertyDetail();
  }, [id]);

  // 转换API数据为组件所需格式
  const transformPropertyDetail = (data) => {
    return data;
  };

  if (loading) {
    return (
      <Layout className="layout">
        <Header style={{ position: 'fixed', zIndex: 1, width: '100%', background: 'white', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
          <div style={{ float: 'left', marginRight: '30px' }}>
            <Typography.Title level={3} style={{ margin: '8px 0', color: '#1890ff' }}>
              <HomeOutlined /> PropertyWize
            </Typography.Title>
          </div>
          <Menu 
            theme="light" 
            mode="horizontal" 
            defaultSelectedKeys={['3']}
            style={{ lineHeight: '64px' }}
            items={[
              { key: '1', label: <Link href="/">首页</Link> },
              { key: '2', label: <Link href="/model-evaluation">模型评估</Link> },
              { key: '3', label: <Link href="/property-reports">房产报告</Link> },
            ]}
          />
        </Header>

        <Content style={{ padding: '0 50px', marginTop: 64 }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            minHeight: 'calc(100vh - 64px - 69px)'
          }}>
            <Spin size="large" tip="加载房产数据中..." />
          </div>
        </Content>

        <Footer style={{ textAlign: 'center' }}>
          PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
        </Footer>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout className="layout">
        <Header style={{ position: 'fixed', zIndex: 1, width: '100%', background: 'white', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
          <div style={{ float: 'left', marginRight: '30px' }}>
            <Typography.Title level={3} style={{ margin: '8px 0', color: '#1890ff' }}>
              <HomeOutlined /> PropertyWize
            </Typography.Title>
          </div>
          <Menu 
            theme="light" 
            mode="horizontal" 
            defaultSelectedKeys={['3']}
            style={{ lineHeight: '64px' }}
            items={[
              { key: '1', label: <Link href="/">首页</Link> },
              { key: '2', label: <Link href="/model-evaluation">模型评估</Link> },
              { key: '3', label: <Link href="/property-reports">房产报告</Link> },
            ]}
          />
        </Header>

        <Content style={{ padding: '0 50px', marginTop: 64 }}>
          <div style={{ 
            background: '#fff', 
            padding: 24, 
            minHeight: 'calc(100vh - 64px - 69px)', 
            borderRadius: '4px', 
            marginTop: '20px' 
          }}>
            <Result
              status="error"
              title="加载失败"
              subTitle={error}
              extra={
                <Button type="primary">
                  <Link href="/property-reports">返回房产报告列表</Link>
                </Button>
              }
            />
          </div>
        </Content>

        <Footer style={{ textAlign: 'center' }}>
          PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
        </Footer>
      </Layout>
    );
  }

  if (!property) {
    return (
      <Layout className="layout">
        <Header style={{ position: 'fixed', zIndex: 1, width: '100%', background: 'white', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
          <div style={{ float: 'left', marginRight: '30px' }}>
            <Typography.Title level={3} style={{ margin: '8px 0', color: '#1890ff' }}>
              <HomeOutlined /> PropertyWize
            </Typography.Title>
          </div>
          <Menu 
            theme="light" 
            mode="horizontal" 
            defaultSelectedKeys={['3']}
            style={{ lineHeight: '64px' }}
            items={[
              { key: '1', label: <Link href="/">首页</Link> },
              { key: '2', label: <Link href="/model-evaluation">模型评估</Link> },
              { key: '3', label: <Link href="/property-reports">房产报告</Link> },
            ]}
          />
        </Header>

        <Content style={{ padding: '0 50px', marginTop: 64 }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            minHeight: 'calc(100vh - 64px - 69px)'
          }}>
            <Result
              status="info"
              title="找不到房产信息"
              subTitle="无法找到指定ID的房产信息"
              extra={
                <Button type="primary">
                  <Link href="/property-reports">返回房产报告列表</Link>
                </Button>
              }
            />
          </div>
        </Content>

        <Footer style={{ textAlign: 'center' }}>
          PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
        </Footer>
      </Layout>
    );
  }

  return (
    <Layout className="layout">
      <Head>
        <title>{property.address || '房产基础信息'} - PropertyWize</title>
        <meta name="description" content={`${property.address || '房产'}的基础信息`} />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Header style={{ position: 'fixed', zIndex: 1, width: '100%', background: 'white', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
        <div style={{ float: 'left', marginRight: '30px' }}>
          <Typography.Title level={3} style={{ margin: '8px 0', color: '#1890ff' }}>
            <HomeOutlined /> PropertyWize
          </Typography.Title>
        </div>
        <Menu 
          theme="light" 
          mode="horizontal" 
          defaultSelectedKeys={['3']}
          style={{ lineHeight: '64px' }}
          items={[
            { key: '1', label: <Link href="/">首页</Link> },
            { key: '2', label: <Link href="/model-evaluation">模型评估</Link> },
            { key: '3', label: <Link href="/property-reports">房产报告</Link> },
          ]}
        />
      </Header>

      <Content style={{ padding: '0 50px', marginTop: 64 }}>
        <div style={{ background: '#fff', padding: 24, borderRadius: '4px', marginTop: '20px' }}>
          {/* 面包屑导航 */}
          <Breadcrumb style={{ marginBottom: '16px' }}>
            <Breadcrumb.Item>
              <Link href="/">首页</Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link href="/property-reports">房产报告</Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link href={`/property-reports/${id}`}>{property.id}</Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>基础信息</Breadcrumb.Item>
          </Breadcrumb>

          {/* 页面标题 */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <Title level={2}>
              <InfoCircleOutlined style={{ marginRight: '8px' }} />
              {property.address} - 基础信息
            </Title>
            <Space>
              <Button type="primary" icon={<ExportOutlined />}>导出PDF报告</Button>
              <Button type="default" icon={<DollarOutlined />}>
                <Link href={`/property-reports/${id}/pricing-analysis`}>查看估价分析</Link>
              </Button>
              <Button type="default" icon={<ArrowLeftOutlined />}>
                <Link href={`/property-reports/${id}`}>返回概览</Link>
              </Button>
            </Space>
          </div>

          {/* 基础信息内容 */}
          <Row>
            <Col xs={24}>
              <PropertyBasicInfo property={property} />
              <PropertyLocation 
                latitude={property.latitude} 
                longitude={property.longitude} 
              />
              <PropertyImages />
            </Col>
          </Row>
        </div>
      </Content>

      <Footer style={{ textAlign: 'center', marginTop: '24px' }}>
        PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
      </Footer>
    </Layout>
  );
} 