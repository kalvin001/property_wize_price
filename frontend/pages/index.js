import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Router from 'next/router';
import { 
  Layout, 
  Menu, 
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
  Empty,
  Drawer,
  Affix,
  Tag
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
  MenuOutlined
} from '@ant-design/icons';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;
const { Search } = Input;

export default function Home() {
  const [searchValue, setSearchValue] = useState('');
  const [searching, setSearching] = useState(false);
  const [featuredProperties, setFeaturedProperties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // 检测移动设备
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => {
      window.removeEventListener('resize', checkMobile);
    };
  }, []);

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

  // 菜单项配置
  const menuItems = [
    { key: '1', label: '首页' },
    { key: '2', label: <Link href="/model-evaluation">模型评估</Link> },
    { key: '3', label: <Link href="/property-reports">房产估价报告</Link> },
  ];

  return (
    <Layout className="layout">
      <Head>
        <title>PropertyWize - 可解释房产估价系统</title>
        <meta name="description" content="基于AI的智能房产估价系统，提供详细的可解释性报告" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Affix>
        <Header style={{ 
          position: 'sticky', 
          zIndex: 1000, 
          width: '100%', 
          background: 'white', 
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
          padding: isMobile ? '0 15px' : '0 50px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Typography.Title level={3} style={{ margin: '8px 0', color: '#1890ff', whiteSpace: 'nowrap' }}>
              <HomeOutlined /> PropertyWize
            </Typography.Title>
          </div>
          
          {isMobile ? (
            <>
              <Button 
                type="text" 
                icon={<MenuOutlined />} 
                onClick={() => setDrawerVisible(true)} 
                size="large"
              />
              <Drawer
                title="导航菜单"
                placement="right"
                onClose={() => setDrawerVisible(false)}
                open={drawerVisible}
                width={250}
              >
                <Menu 
                  mode="vertical" 
                  defaultSelectedKeys={['1']}
                  items={menuItems}
                  onClick={() => setDrawerVisible(false)}
                />
              </Drawer>
            </>
          ) : (
            <Menu 
              theme="light" 
              mode="horizontal" 
              defaultSelectedKeys={['1']}
              style={{ lineHeight: '64px', border: 'none' }}
              items={menuItems}
            />
          )}
        </Header>
      </Affix>

      <Content style={{ padding: isMobile ? '0 15px' : '0 50px', marginTop: 16 }}>
        {/* 1. 简化宣传横幅 */}
        <div className="hero-banner" style={{ 
          height: isMobile ? 'auto' : '240px', 
          padding: isMobile ? '40px 15px' : '0',
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
            position: 'relative',
            width: '100%',
            padding: '0 15px',
            zIndex: 2,
            color: 'white',
            textAlign: 'center'
          }}>
            <Title style={{ 
              color: 'white', 
              fontSize: isMobile ? '28px' : '42px', 
              marginBottom: '10px',
              wordBreak: 'break-word'
            }}>
              智能房产估价 · 精准决策支持
            </Title>
            <Paragraph style={{ 
              color: 'white', 
              fontSize: isMobile ? '14px' : '16px', 
              marginBottom: '20px' 
            }}>
              PropertyWize基于AI和大数据，为您提供准确的房产估价与详细的解释报告
            </Paragraph>
            <Space size={isMobile ? 'small' : 'middle'} wrap={isMobile}>
              <Button type="primary" size={isMobile ? 'middle' : 'large'} onClick={() => {
                const element = document.getElementById('search-section');
                if (element) element.scrollIntoView({ behavior: 'smooth' });
              }}>开始估价</Button>
              <Button ghost size={isMobile ? 'middle' : 'large'} style={{ color: 'white', borderColor: 'white' }}>
                <Link href="/property-reports" style={{ color: 'white' }}>查看房产估价报告</Link>
              </Button>
            </Space>
          </div>
        </div>

        {/* 3. 搜索区域 */}
        <div id="search-section" style={{ 
          textAlign: 'center', 
          margin: isMobile ? '40px 0' : '60px 0', 
          background: '#f9f9f9', 
          padding: isMobile ? '20px 15px' : '30px', 
          borderRadius: '8px' 
        }}>
          <Title level={2}>查询房产估价</Title>
          <Paragraph style={{ 
            fontSize: isMobile ? '14px' : '16px', 
            marginBottom: isMobile ? '20px' : '30px' 
          }}>
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
        <div style={{ margin: isMobile ? '40px 0' : '60px 0' }}>
          <Title level={2} style={{ textAlign: 'center', marginBottom: isMobile ? '15px' : '20px' }}>精选房产估价案例</Title>
          <Paragraph style={{ 
            textAlign: 'center', 
            fontSize: isMobile ? '14px' : '16px', 
            marginBottom: isMobile ? '25px' : '40px',
            padding: '0 10px'
          }}>
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
            <Row gutter={[16, 16]}>
              {featuredProperties.map(property => (
                <Col xs={24} sm={12} md={8} key={property.id} className="property-card-mobile">
                  <Card
                    hoverable
                    style={{ height: '100%' }}
                    cover={
                      <div style={{
                        height: isMobile ? '180px' : '200px',
                        background: '#f0f2f5',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <HomeOutlined style={{ fontSize: '48px', color: '#bfbfbf' }} />
                      </div>
                    }
                    onClick={() => Router.push(`/property-reports/${property.id}`)}
                  >
                    <Card.Meta
                      title={property.address}
                      description={
                        <Space direction="vertical" style={{ width: '100%', marginTop: '8px' }}>
                          <Statistic
                            title="AI估价"
                            value={property.predicted_price}
                            precision={0}
                            formatter={value => `A$${(value / 10000).toFixed(0)}万`}
                            valueStyle={{ 
                              color: '#1890ff', 
                              fontSize: isMobile ? '18px' : '20px' 
                            }}
                          />
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
                            {property.beds > 0 && (
                              <Tag color="blue">{property.beds} 卧室</Tag>
                            )}
                            {property.baths > 0 && (
                              <Tag color="green">{property.baths} 浴室</Tag>
                            )}
                            {property.area > 0 && (
                              <Tag color="purple">{property.area}㎡</Tag>
                            )}
                          </div>
                        </Space>
                      }
                    />
                  </Card>
                </Col>
              ))}
            </Row>
          )}
        </div>
      </Content>

      <Footer style={{ textAlign: 'center', padding: isMobile ? '20px 15px' : '24px 50px' }}>
        PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
      </Footer>
    </Layout>
  );
} 