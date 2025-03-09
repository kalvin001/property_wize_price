import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { 
  Layout, 
  Menu, 
  Typography, 
  Card, 
  Row, 
  Col, 
  Button, 
  Input, 
  Divider, 
  Tag, 
  Spin,
  Empty,
  Statistic,
  List,
  Avatar,
  Pagination,
  notification
} from 'antd';
import { 
  HomeOutlined, 
  BarChartOutlined, 
  FileTextOutlined,
  SearchOutlined,
  EnvironmentOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  DollarOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;
const { Search } = Input;

export default function PropertyReports() {
  const router = useRouter();
  const { search } = router.query;
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchValue, setSearchValue] = useState('');
  const [properties, setProperties] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(9);

  // 处理URL参数
  useEffect(() => {
    // 如果有搜索参数，设置搜索值
    if (search) {
      setSearchValue(search);
    }
  }, [search]);

  // 获取房产列表
  useEffect(() => {
    // 首次加载时使用URL参数
    fetchProperties(1, pageSize, search || null);
  }, []);

  // 当URL参数变化时重新获取数据
  useEffect(() => {
    if (router.isReady) {
      fetchProperties(1, pageSize, search || null);
    }
  }, [router.isReady, search]);

  // 获取房产列表
  const fetchProperties = async (page = 1, size = 9, query = null) => {
    setLoading(true);
    setError(null);
    
    try {
      // 构建API请求URL
      let url = `/api/properties?page=${page}&page_size=${size}`;
      if (query) {
        url += `&query=${encodeURIComponent(query)}`;
      }
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`请求失败: ${response.status}`);
      }
      
      const data = await response.json();
      
      setProperties(data.properties.map(prop => transformProperty(prop)));
      setTotalCount(data.total);
      setCurrentPage(data.page);
    } catch (err) {
      console.error('获取房产数据失败:', err);
      setError('加载房产数据失败，请稍后再试。');
      notification.error({
        message: '数据加载失败',
        description: err.message || '获取房产列表出错'
      });
    } finally {
      setLoading(false);
    }
  };
  
  // 转换房产数据为前端展示格式
  const transformProperty = (prop) => {
    // 提取常用特征
    const features = prop.features || {};
    
    // 获取实际价格和预测价格
    const actualPrice = features.y_label || 0;
    const predictedPrice = prop.predicted_price || 0;
    
    // 计算误差百分比
    let errorPercent = 0;
    if (actualPrice > 0 && predictedPrice > 0) {
      errorPercent = ((predictedPrice - actualPrice) / actualPrice * 100).toFixed(1);
    }
    
    return {
      id: prop.prop_id,
      address: prop.address,
      type: features.prop_type || '未知类型',
      predicted_price: predictedPrice,
      actual_price: actualPrice,
      error_percent: errorPercent,
      beds: features.prop_bed || 0,
      baths: features.prop_bath || 0,
      area: features.prop_area || features.land_size || 0,
      key_features: extractKeyFeatures(features),
      updated_at: features.updated_date ? new Date(features.updated_date).toLocaleDateString() : '未知'
    };
  };
  
  // 从特征中提取关键特点
  const extractKeyFeatures = (features) => {
    const keyFeatures = [];
    
    // 判断各种特征并添加关键词
    if (features.prop_type) keyFeatures.push(features.prop_type);
    if (features.locality_name) keyFeatures.push(features.locality_name);
    if (features.prop_carpark && features.prop_carpark > 0) keyFeatures.push(`${features.prop_carpark}车位`);
    if (features.prop_area && features.land_size && features.prop_area < features.land_size) keyFeatures.push('带花园');
    if (features.prop_zoning) keyFeatures.push(`${features.prop_zoning}区`);
    if (features.waterfront_type === 1) keyFeatures.push('临水');
    if (features.prop_postcode) keyFeatures.push(`邮编${features.prop_postcode}`);
    
    // 根据房龄添加特点
    if (features.prop_build_year) {
      const age = new Date().getFullYear() - features.prop_build_year;
      if (age <= 5) keyFeatures.push('新房');
      else if (age <= 10) keyFeatures.push('较新');
    }
    
    return keyFeatures.slice(0, 5); // 最多返回5个特点
  };

  // 搜索房产
  const searchProperties = (value) => {
    setSearchValue(value);
    setCurrentPage(1); // 重置到第一页
    
    // 更新URL参数
    const query = {};
    if (value) query.search = value;
    
    router.push({
      pathname: '/property-reports',
      query
    }, undefined, { shallow: true });
    
    fetchProperties(1, pageSize, value || null);
  };
  
  // 处理分页变化
  const handlePageChange = (page, pageSize) => {
    setCurrentPage(page);
    fetchProperties(page, pageSize, searchValue || null);
  };

  return (
    <Layout className="layout">
      <Head>
        <title>房产估价报告 - PropertyWize</title>
        <meta name="description" content="浏览房产估价报告样例，了解AI如何分析不同类型的房产" />
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
            { key: '3', label: '房产估价报告' },
          ]}
        />
      </Header>

      <Content style={{ padding: '0 50px', marginTop: 64 }}>
        <div style={{ background: '#fff', padding: 24, minHeight: 'calc(100vh - 64px - 69px)', borderRadius: '4px', marginTop: '20px' }}>
          <Title level={2}>房产估价报告</Title>
          <Paragraph>
            浏览AI分析的真实房产估价报告，了解各类房产的价值评估及价格影响因素。
          </Paragraph>

          {/* 搜索区域 */}
          <div style={{ marginBottom: '24px' }}>
            <Row gutter={16}>
              <Col xs={24} sm={12} md={8} lg={6}>
                <Search
                  placeholder="搜索房产地址或ID"
                  onSearch={searchProperties}
                  onChange={(e) => setSearchValue(e.target.value)}
                  value={searchValue}
                  style={{ width: '100%' }}
                  allowClear
                />
              </Col>
              <Col xs={24} sm={12} md={8} lg={6}>
                <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
                  <Text type="secondary">共找到 {totalCount} 条记录</Text>
                </div>
              </Col>
            </Row>
          </div>

          {/* 显示小标题 */}
          <Divider orientation="left">房产列表</Divider>

          {/* 房产估价报告列表 */}
          {loading ? (
            <div style={{ textAlign: 'center', margin: '50px 0' }}>
              <Spin size="large" />
              <p style={{ marginTop: 20 }}>加载房产数据...</p>
            </div>
          ) : error ? (
            <div style={{ textAlign: 'center', margin: '50px 0' }}>
              <InfoCircleOutlined style={{ fontSize: 32, color: '#ff4d4f' }} />
              <p style={{ marginTop: 20 }}>{error}</p>
              <Button 
                type="primary" 
                onClick={() => fetchProperties(currentPage, pageSize, searchValue)}
                style={{ marginTop: 16 }}
              >
                重试
              </Button>
            </div>
          ) : properties.length === 0 ? (
            <Empty description="没有找到符合条件的房产估价报告" />
          ) : (
            <>
              <List
                grid={{ gutter: 16, xs: 1, sm: 1, md: 2, lg: 3, xl: 3, xxl: 3 }}
                dataSource={properties}
                renderItem={property => (
                  <List.Item>
                    <Card 
                      hoverable 
                      style={{ height: '100%' }}
                      actions={[
                        <Link key="view" href={`/property-reports/${property.id}`}>
                          <Button type="primary">查看详情</Button>
                        </Link>
                      ]}
                    >
                      <div style={{ display: 'flex', marginBottom: '16px' }}>
                        <Avatar 
                          size={64} 
                          icon={<HomeOutlined />} 
                          style={{ 
                            backgroundColor: '#1890ff',
                            marginRight: '16px'
                          }}
                        />
                        <div>
                          <Title level={4} style={{ margin: 0, marginBottom: '8px' }}>{property.id}</Title>
                          {property.type && (
                            <Tag color="blue">
                              {property.type}
                            </Tag>
                          )}
                        </div>
                      </div>

                      <Paragraph ellipsis={{ rows: 2 }}>
                        <EnvironmentOutlined style={{ marginRight: '8px' }} />
                        {property.address}
                      </Paragraph>

                      <Row gutter={16}>
                        <Col span={12}>
                          <Statistic 
                            title="估价" 
                            value={property.predicted_price} 
                            precision={0} 
                            formatter={value => `A$${(value / 10000).toFixed(0)}万`}
                            valueStyle={{ color: '#1890ff' }}
                          />
                        </Col>
                        {property.actual_price > 0 && (
                          <Col span={12}>
                            <Statistic 
                              title="误差" 
                              value={property.error_percent} 
                              precision={1} 
                              prefix={property.error_percent > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                              suffix="%" 
                              valueStyle={{ color: Math.abs(property.error_percent) < 5 ? '#3f8600' : '#cf1322' }}
                            />
                          </Col>
                        )}
                      </Row>

                      <Divider style={{ margin: '16px 0' }} />

                      <Row gutter={[16, 16]}>
                        {property.beds > 0 && (
                          <Col span={8}>
                            <div style={{ textAlign: 'center' }}>
                              <div>{property.beds}室</div>
                            </div>
                          </Col>
                        )}
                        {property.baths > 0 && (
                          <Col span={8}>
                            <div style={{ textAlign: 'center' }}>
                              <div>{property.baths}卫</div>
                            </div>
                          </Col>
                        )}
                        {property.area > 0 && (
                          <Col span={8}>
                            <div style={{ textAlign: 'center' }}>
                              <div>{property.area}m²</div>
                            </div>
                          </Col>
                        )}
                      </Row>

                      <div style={{ marginTop: '16px' }}>
                        {property.key_features.map((feature, index) => (
                          <Tag key={index} style={{ marginBottom: '8px' }}>{feature}</Tag>
                        ))}
                      </div>
                    </Card>
                  </List.Item>
                )}
              />
              
              {/* 分页控件 */}
              {totalCount > pageSize && (
                <div style={{ textAlign: 'center', marginTop: '30px' }}>
                  <Pagination 
                    current={currentPage}
                    total={totalCount}
                    pageSize={pageSize}
                    onChange={handlePageChange}
                    showSizeChanger={false}
                    showQuickJumper
                  />
                </div>
              )}
            </>
          )}
        </div>
      </Content>

      <Footer style={{ textAlign: 'center' }}>
        PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
      </Footer>
    </Layout>
  );
} 