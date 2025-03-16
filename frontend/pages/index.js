import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Router from 'next/router';
import { 
  Typography, 
  Button, 
  Input, 
  Row, 
  Col, 
  Card, 
  Carousel, 
  Statistic, 
  Divider,
  Space,
  message,
  Spin,
  Empty
} from 'antd';
import { 
  SearchOutlined, 
  HomeOutlined, 
  BarChartOutlined, 
  FileTextOutlined,
  CheckCircleOutlined,
  RocketOutlined,
  BulbOutlined,
  BookOutlined,
  EnvironmentOutlined,
  DollarOutlined,
  SettingOutlined,
  AppstoreOutlined
} from '@ant-design/icons';
import AppLayout from '../components/Layout/AppLayout';

const { Title, Paragraph, Text } = Typography;
const { Search } = Input;

export default function Home() {
  const [searchValue, setSearchValue] = useState('');
  const [searching, setSearching] = useState(false);
  const [featuredProperties, setFeaturedProperties] = useState([]);
  const [loading, setLoading] = useState(true);

  // 获取精选房产示例
  useEffect(() => {
    const fetchFeaturedProperties = async () => {
      setLoading(true);
      try {
        // 随机获取3个房产
        const response = await fetch('/api/properties?page=1&page_size=3');
        if (!response.ok) {
          throw new Error('获取房产数据失败');
        }
        
        const data = await response.json();
        if (data.properties && data.properties.length > 0) {
          setFeaturedProperties(data.properties.map(formatProperty));
        }
      } catch (error) {
        console.error('获取精选房产失败:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchFeaturedProperties();
  }, []);
  
  // 格式化房产数据
  const formatProperty = (property) => {
    const features = property.features || {};
    
    // 获取实际价格和预测价格
    const actualPrice = features.y_label || 0;
    const predictedPrice = property.predicted_price || 0;
    
    return {
      id: property.prop_id,
      address: property.address,
      predicted_price: predictedPrice,
      actual_price: actualPrice,
      area: features.prop_area || features.land_size || 0,
      beds: features.prop_bed || 0,
      baths: features.prop_bath || 0,
      type: features.prop_type || '',
      locality: features.locality_name || ''
    };
  };

  // 处理搜索
  const handleSearch = async (value) => {
    if (!value || value.trim() === '') {
      message.warning('请输入房产地址或ID');
      return;
    }

    setSearching(true);

    try {
      // 尝试查询ID直接跳转
      if (value.toLowerCase().startsWith('prop-') || /^\d+$/.test(value)) {
        // 如果像是ID格式，直接尝试跳转到详情页
        const propId = value.toLowerCase().startsWith('prop-') ? value : `prop-${value}`;
        Router.push(`/property-reports/${propId}`);
        return;
      }

      // 否则执行搜索请求
      const response = await fetch(`/api/properties?query=${encodeURIComponent(value)}&page_size=10`);
      
      if (!response.ok) {
        throw new Error('搜索请求失败');
      }
      
      const data = await response.json();
      
      if (data.properties && data.properties.length > 0) {
        // 有结果，重定向到房产估价报告页面，保留搜索条件
        Router.push({
          pathname: '/property-reports',
          query: { search: value }
        });
      } else {
        // 无结果，也重定向到房产估价报告页面
        message.info('未找到匹配的房产，显示所有房产');
        Router.push('/property-reports');
      }
    } catch (error) {
      console.error('搜索出错:', error);
      message.error('搜索失败，请稍后再试');
    } finally {
      setSearching(false);
    }
  };

  return (
    <AppLayout>
      <Head>
        <title>PropertyWize - 可解释房产估价系统</title>
        <meta name="description" content="基于AI的智能房产估价系统，提供详细的可解释性报告" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* 1. 简化宣传横幅 */}
      <div className="hero-banner" style={{ 
        height: '240px', 
        background: 'linear-gradient(135deg, #1890ff 0%, #001529 100%)',
        display: 'flex',
        alignItems: 'center',
        borderRadius: '8px',
        marginTop: '30px',
        overflow: 'hidden',
        position: 'relative',
      }}>
        <div style={{ 
          position: 'absolute', 
          width: '100%', 
          height: '100%', 
          backgroundColor: 'rgba(0,0,0,0.3)',
          zIndex: 1
        }}></div>
        <div style={{ 
          position: 'absolute',
          width: '80%',
          left: '10%',
          zIndex: 2,
          color: 'white',
          textAlign: 'center'
        }}>
          <Title style={{ color: 'white', fontSize: '42px', marginBottom: '10px' }}>
            智能房产估价 · 精准决策支持
          </Title>
          <Paragraph style={{ color: 'white', fontSize: '16px', marginBottom: '20px' }}>
            PropertyWize基于AI和大数据，为您提供准确的房产估价与详细的解释报告
          </Paragraph>
          <Space size="middle">
            <Button type="primary" size="large" onClick={() => {
              const element = document.getElementById('search-section');
              if (element) element.scrollIntoView({ behavior: 'smooth' });
            }}>开始估价</Button>
            <Button ghost size="large" style={{ color: 'white', borderColor: 'white' }}>
              <Link href="/property-reports" style={{ color: 'white' }} legacyBehavior>查看房产估价报告</Link>
            </Button>
          </Space>
        </div>
      </div>

      {/* 3. 搜索区域 */}
      <div id="search-section" style={{ textAlign: 'center', margin: '60px 0', background: '#f9f9f9', padding: '30px', borderRadius: '8px' }}>
        <Title level={2}>查询房产估价</Title>
        <Paragraph style={{ fontSize: '16px', marginBottom: '30px' }}>
          输入房产地址或ID，查询现有报告或生成新的估价报告
        </Paragraph>
        <Row justify="center">
          <Col xs={24} sm={24} md={16} lg={12} xl={8}>
            <Search
              placeholder="输入房产地址或ID"
              enterButton="查询估价"
              size="large"
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              onSearch={handleSearch}
              loading={searching}
              style={{ width: '100%' }}
            />
          </Col>
        </Row>
      </div>

      {/* 2. 真实房产展示 */}
      <div style={{ margin: '60px 0' }}>
        <Title level={2} style={{ textAlign: 'center', marginBottom: '20px' }}>精选房产估价案例</Title>
        <Paragraph style={{ textAlign: 'center', fontSize: '16px', marginBottom: '40px' }}>
          以下展示的是真实房产的AI估价案例，点击查看详细的价格影响因素分析
        </Paragraph>
        
        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Spin size="large" />
            <p style={{ marginTop: 16 }}>加载房产数据...</p>
          </div>
        ) : featuredProperties.length === 0 ? (
          <Empty description="暂无房产案例数据" />
        ) : (
          <Row gutter={[24, 24]}>
            {featuredProperties.map(property => (
              <Col xs={24} sm={12} md={8} key={property.id}>
                <Card
                  hoverable
                  cover={<div style={{ 
                    height: '200px', 
                    background: '#f0f5ff',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}>
                    <FileTextOutlined style={{ fontSize: '64px', color: '#1890ff' }} />
                  </div>}
                >
                  <Card.Meta
                    title={
                      <div style={{whiteSpace: 'normal', height: '48px', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                        {property.address}
                      </div>
                    }
                    description={
                      <div>
                        <div style={{ marginBottom: '8px' }}>
                          <EnvironmentOutlined style={{ marginRight: '4px' }} />
                          {property.locality || '未知区域'}
                          {property.type && ` | ${property.type}`}
                        </div>
                        <div>
                          {property.area > 0 && `${property.area}m² | `}
                          {property.beds > 0 && `${property.beds}室`}
                          {property.baths > 0 && `${property.baths}卫`}
                        </div>
                        <div style={{ marginTop: '8px', fontWeight: 'bold', color: '#1890ff' }}>
                          <DollarOutlined style={{ marginRight: '4px' }} />
                          估价: A${(property.predicted_price / 10000).toFixed(0)}万
                        </div>
                      </div>
                    }
                  />
                  <Link href={`/property-reports/${property.id}`} legacyBehavior>
                    <Button type="link" style={{ padding: 0, marginTop: '15px' }}>
                      查看详情 →
                    </Button>
                  </Link>
                </Card>
              </Col>
            ))}
          </Row>
        )}
        
        <div style={{ textAlign: 'center', marginTop: '30px' }}>
          <Link href="/property-reports" legacyBehavior>
            <Button type="primary">查看更多房产估价报告</Button>
          </Link>
        </div>
      </div>

      {/* 4. 我们的优势 */}
      <div style={{ margin: '60px 0' }}>
        <Title level={2} style={{ textAlign: 'center', marginBottom: '40px' }}>为什么选择 PropertyWize</Title>
        
        <Row gutter={[32, 32]}>
          <Col xs={24} sm={12} md={8}>
            <Card 
              hoverable 
              style={{ height: '100%', textAlign: 'center', padding: '20px' }}
            >
              <BulbOutlined style={{ fontSize: '48px', color: '#1890ff', margin: '20px 0' }} />
              <Card.Meta
                title="AI驱动的准确估价"
                description="基于先进的机器学习算法和海量数据，提供准确的房产价值估算"
              />
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={8}>
            <Card 
              hoverable 
              style={{ height: '100%', textAlign: 'center', padding: '20px' }}
            >
              <BarChartOutlined style={{ fontSize: '48px', color: '#1890ff', margin: '20px 0' }} />
              <Card.Meta
                title="详细的可解释性分析"
                description="透明展示影响房价的各项因素及其权重，让您了解每一分钱的来源"
              />
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={8}>
            <Card 
              hoverable 
              style={{ height: '100%', textAlign: 'center', padding: '20px' }}
            >
              <FileTextOutlined style={{ fontSize: '48px', color: '#1890ff', margin: '20px 0' }} />
              <Card.Meta
                title="专业的估价报告"
                description="生成全面专业的PDF报告，包含详尽的房产分析和市场比较数据"
              />
            </Card>
          </Col>
        </Row>
      </div>

      {/* 5. 数据统计 */}
      <div style={{ 
        margin: '60px 0', 
        padding: '40px 0',
        background: '#f0f5ff',
        borderRadius: '8px',
        textAlign: 'center'
      }}>
        <Title level={2} style={{ marginBottom: '40px' }}>可信赖的房产估价服务</Title>
        
        <Row gutter={[48, 48]} justify="center">
          <Col xs={24} sm={12} md={6}>
            <Statistic title="估价准确率" value={96.8} suffix="%" />
            <Paragraph>高于行业平均水平</Paragraph>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Statistic title="已处理房产" value={125000} />
            <Paragraph>来自全国各地的房产数据</Paragraph>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Statistic title="考虑因素" value={50} suffix="+" />
            <Paragraph>综合分析影响房价的因素</Paragraph>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Statistic title="客户满意度" value={98.2} suffix="%" />
            <Paragraph>基于10000+用户反馈</Paragraph>
          </Col>
        </Row>
      </div>

      {/* 6. 行动召唤 */}
      <div style={{ 
        margin: '60px 0', 
        padding: '40px',
        background: 'linear-gradient(135deg, #f6ffed 0%, #e6f7ff 100%)',
        borderRadius: '8px',
        textAlign: 'center'
      }}>
        <Title level={2}>开始您的房产估价</Title>
        <Paragraph style={{ fontSize: '16px', maxWidth: '700px', margin: '20px auto 30px' }}>
          无论您是准备购买、出售房产，还是希望了解当前房产的市场价值，
          PropertyWize都能为您提供准确、透明的房产估价服务。
        </Paragraph>
        <Button type="primary" size="large" onClick={() => {
          const element = document.getElementById('search-section');
          if (element) element.scrollIntoView({ behavior: 'smooth' });
        }}>立即开始</Button>
      </div>

      {/* 功能卡片区 */}
      <Row gutter={[24, 24]} style={{ marginTop: 48, marginBottom: 48 }}>
        <Col xs={24} sm={12} md={8}>
          <Card 
            hoverable 
            style={{ height: '100%' }}
            onClick={() => Router.push('/model-center')}
          >
            <Space direction="vertical" size="middle" style={{display: 'flex', alignItems: 'center'}}>
              <AppstoreOutlined style={{ fontSize: 48, color: '#1890ff' }} />
              <Title level={4}>模型中心</Title>
              <Text>查看模型性能指标，管理和训练不同类型的房价预测模型</Text>
            </Space>
          </Card>
        </Col>

        <Col xs={24} sm={12} md={8}>
          <Card 
            hoverable 
            style={{ height: '100%' }}
            onClick={() => Router.push('/property-reports')}
          >
            <Space direction="vertical" size="middle" style={{display: 'flex', alignItems: 'center'}}>
              <FileTextOutlined style={{ fontSize: 48, color: '#52c41a' }} />
              <Title level={4}>房产估价报告</Title>
              <Text>查看详细的房产估价报告和可视化分析</Text>
            </Space>
          </Card>
        </Col>

        <Col xs={24} sm={12} md={8}>
          <Card 
            hoverable 
            style={{ height: '100%' }}
            onClick={() => {
              const element = document.getElementById('search-section');
              if (element) element.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            <Space direction="vertical" size="middle" style={{display: 'flex', alignItems: 'center'}}>
              <SearchOutlined style={{ fontSize: 48, color: '#722ed1' }} />
              <Title level={4}>查询估价</Title>
              <Text>输入房产地址或ID，查询或生成新的估价报告</Text>
            </Space>
          </Card>
        </Col>
      </Row>
    </AppLayout>
  );
} 