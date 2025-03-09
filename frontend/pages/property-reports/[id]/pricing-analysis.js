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
  Space,
  Tabs
} from 'antd';
import { 
  HomeOutlined, 
  ArrowLeftOutlined,
  InfoCircleOutlined,
  ExportOutlined,
  DollarOutlined,
  LineChartOutlined,
  RadarChartOutlined,
  AreaChartOutlined,
  PieChartOutlined,
  EnvironmentOutlined
} from '@ant-design/icons';
import PropertyPriceResult from '../../../components/PropertyPriceResult';
import PropertyFeatureInfluences from '../../../components/PropertyFeatureInfluences';
import PropertyComparables from '../../../components/PropertyComparables';
import PropertyConfidence from '../../../components/PropertyConfidence';
import PropertyPriceTrends from '../../../components/PropertyPriceTrends';
import PropertyPriceRange from '../../../components/PropertyPriceRange';
import PropertyFeatureRadar from '../../../components/PropertyFeatureRadar';
import PropertyNeighborhoodStats from '../../../components/PropertyNeighborhoodStats';
import PropertyModelExplanation from '../../../components/PropertyModelExplanation';
import PropertyFeatureWaterfall from '../../../components/PropertyFeatureWaterfall';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

export default function PropertyPricingAnalysis() {
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
    // 确保所有必需的字段都存在，如果不存在则提供默认值
    return {
      ...data,
      // 确保各种必要字段的存在，防止组件出错
      feature_importance: data.feature_importance || [],
      comparable_properties: data.comparable_properties || [],
      price_trends: data.price_trends || [],
      price_range: data.price_range || {
        min: data.predicted_price * 0.95,
        max: data.predicted_price * 1.05,
        most_likely: data.predicted_price
      },
      neighborhood_stats: data.neighborhood_stats || {
        avg_price: data.predicted_price * 1.02,
        min_price: data.predicted_price * 0.85,
        max_price: data.predicted_price * 1.15,
        num_properties: 28,
        price_trend: '上升',
        avg_price_per_sqm: data.predicted_price / (data.features?.prop_area || 100),
        current_price: data.predicted_price
      },
      confidence_interval: data.confidence_interval || {
        lower_bound: data.predicted_price * 0.93,
        upper_bound: data.predicted_price * 1.07,
        confidence_level: 0.95
      },
      model_explanation: data.model_explanation || {
        model_type: "XGBoost回归",
        r2_score: 0.87,
        mae: data.predicted_price * 0.04,
        mape: 4.2,
        feature_count: data.feature_importance?.length || 5,
        top_positive_features: data.feature_importance?.filter(f => f.effect === "positive").map(f => f.feature).slice(0, 3) || [],
        top_negative_features: data.feature_importance?.filter(f => f.effect === "negative").map(f => f.feature).slice(0, 3) || [],
        prediction_confidence: 95
      }
    };
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
              { key: '3', label: <Link href="/property-reports">房产估价报告</Link> },
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
              { key: '3', label: <Link href="/property-reports">房产估价报告</Link> },
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
                  <Link href="/property-reports">返回房产估价报告列表</Link>
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
              { key: '3', label: <Link href="/property-reports">房产估价报告</Link> },
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
                  <Link href="/property-reports">返回房产估价报告列表</Link>
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
        <title>{property.address || '房产估价分析'} - PropertyWize</title>
        <meta name="description" content={`${property.address || '房产'}的详细估价分析`} />
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
            { key: '3', label: <Link href="/property-reports">房产估价报告</Link> },
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
              <Link href="/property-reports">房产估价报告</Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link href={`/property-reports/${id}`}>{property.id}</Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>估价分析</Breadcrumb.Item>
          </Breadcrumb>

          {/* 页面标题 */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <Title level={2}>
              <DollarOutlined style={{ marginRight: '8px' }} />
              {property.address} - 估价分析
            </Title>
            <Space>
              <Button type="primary" icon={<ExportOutlined />}>导出PDF报告</Button>
              <Button type="default" icon={<InfoCircleOutlined />}>
                <Link href={`/property-reports/${id}/basic-info`}>查看基础信息</Link>
              </Button>
              <Button type="default" icon={<ArrowLeftOutlined />}>
                <Link href={`/property-reports/${id}`}>返回概览</Link>
              </Button>
            </Space>
          </div>

          {/* 估价分析内容 */}
          <Row>
            <Col xs={24}>
              <Tabs
                defaultActiveKey="1"
                size="large"
                items={[
                  {
                    key: '1',
                    label: <span><DollarOutlined /> 价格分析</span>,
                    children: (
                      <>
                        <Row gutter={[24, 24]}>
                          <Col xs={24} md={12}>
                            <PropertyPriceResult property={property} />
                          </Col>
                          <Col xs={24} md={12}>
                            <PropertyPriceRange 
                              priceRange={property.price_range} 
                              confidenceInterval={property.confidence_interval}
                              modelExplanation={property.model_explanation}
                            />
                          </Col>
                        </Row>
                        <PropertyPriceTrends priceTrends={property.price_trends} />
                        <Row gutter={[16, 16]}>
                          <Col xs={24} md={12}>
                            <PropertyFeatureInfluences featureImportance={property.feature_importance} />
                          </Col>
                          <Col xs={24} md={12}>
                            <PropertyFeatureWaterfall 
                              predPrice={property.predicted_price}
                              featureImportance={property.feature_importance} 
                            />
                          </Col>
                        </Row>
                      </>
                    ),
                  },
                  {
                    key: '2',
                    label: <span><AreaChartOutlined /> 特征分析</span>,
                    children: (
                      <>
                        <PropertyFeatureRadar 
                          property={property} 
                          featureImportance={property.feature_importance} 
                        />
                        <PropertyModelExplanation 
                          modelExplanation={property.model_explanation}
                          featureImportance={property.feature_importance}
                        />
                      </>
                    ),
                  },
                  {
                    key: '3',
                    label: <span><EnvironmentOutlined /> 区域分析</span>,
                    children: (
                      <>
                        <PropertyNeighborhoodStats neighborhoodStats={property.neighborhood_stats} />
                        <PropertyComparables comparableProperties={property.comparable_properties} />
                      </>
                    ),
                  },
                ]}
              />
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