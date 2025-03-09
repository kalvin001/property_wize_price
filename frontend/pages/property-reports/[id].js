import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Link from 'next/link';
import { 
  Layout, 
  Menu, 
  Typography, 
  Card, 
  Row, 
  Col, 
  Button, 
  Descriptions, 
  Tag, 
  Divider, 
  Statistic, 
  Table,
  Progress,
  Breadcrumb,
  Spin,
  Result,
  Space,
  Image,
  Empty,
  notification,
  Tabs
} from 'antd';
import { 
  HomeOutlined, 
  BarChartOutlined, 
  ArrowLeftOutlined,
  EnvironmentOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  RiseOutlined,
  FallOutlined,
  InfoCircleOutlined,
  DollarOutlined,
  ExportOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import PropertyLocation from '../../components/PropertyLocation';
import PropertyImages from '../../components/PropertyImages';
import PropertyBasicInfo from '../../components/PropertyBasicInfo';
import PropertyPriceResult from '../../components/PropertyPriceResult';
import PropertyComparables from '../../components/PropertyComparables';
import PropertyFeatureInfluences from '../../components/PropertyFeatureInfluences';
import PropertyConfidence from '../../components/PropertyConfidence';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;

export default function PropertyDetail() {
  const router = useRouter();
  const { id } = router.query;
  const [property, setProperty] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('1');

  // 获取房产详情
  useEffect(() => {
    if (!id) return;
    
    const fetchPropertyDetail = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`/api/properties/${id}`);
        
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('找不到该房产信息');
          } else {
            throw new Error(`请求失败: ${response.status}`);
          }
        }
        
        const data = await response.json();
        setProperty(transformPropertyDetail(data));
      } catch (err) {
        console.error('获取房产详情失败:', err);
        setError(err.message || '加载房产详情失败');
        notification.error({
          message: '数据加载失败',
          description: err.message || '获取房产详情出错'
        });
      } finally {
        setLoading(false);
      }
    };
    
    fetchPropertyDetail();
  }, [id]);

  // 获取URL中的tab参数并设置激活的标签
  useEffect(() => {
    if (router.query.tab === 'basic-info') {
      setActiveTab('2');
    } else {
      setActiveTab('1'); // 默认显示估价分析
    }
  }, [router.query.tab]);

  // 处理标签页变化
  const handleTabChange = (key) => {
    setActiveTab(key);
    // 更新URL但不刷新页面
    router.push(
      {
        pathname: `/property-reports/${id}`,
        query: { tab: key === '2' ? 'basic-info' : 'pricing-analysis' },
      },
      undefined,
      { shallow: true }
    );
  };

  // 转换房产详情数据
  const transformPropertyDetail = (data) => {
    if (!data) return null;
    
    const features = data.features || {};
    
    // 获取实际价格和预测价格
    const actualPrice = features.y_label || 0;
    const predictedPrice = data.predicted_price || 0;
    
    // 计算误差百分比
    let errorPercent = 0;
    if (actualPrice > 0 && predictedPrice > 0) {
      errorPercent = ((predictedPrice - actualPrice) / actualPrice * 100).toFixed(1);
    }
    
    // 提取房产特点
    const keyFeatures = [];
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
    
    return {
      id: data.prop_id,
      address: data.address,
      type: features.prop_type || '未知类型',
      predicted_price: predictedPrice,
      actual_price: actualPrice,
      error_percent: errorPercent,
      region: features.locality_name || features.region || '未知区域',
      community: features.street_name || '未知社区',
      beds: features.prop_bed || 0,
      baths: features.prop_bath || 0,
      area: features.prop_area || features.land_size || 0,
      floor: features.prop_floor || 1,
      total_floors: features.floors_total || features.prop_floor + 2 || 1,
      orientation: features.orientation || '南北',
      decoration: features.decoration || '普通装修',
      year_built: features.prop_build_year || new Date().getFullYear() - 10,
      updated_at: features.updated_date ? new Date(features.updated_date).toLocaleDateString() : '未知',
      key_features: keyFeatures,
      feature_importance: data.feature_importance || [],
      comparable_properties: data.comparable_properties || [],
      
      // 添加更多特征
      land_size: features.land_size || 0,
      carpark: features.prop_carpark || 0,
      longitude: features.prop_x || 0,
      latitude: features.prop_y || 0,
      postcode: features.locality_post || features.prop_postcode || '未知',
      // 更多可能的特征...
    };
  };

  // 特征重要性表格列
  const featureColumns = [
    {
      title: '特征名称',
      dataIndex: 'feature',
      key: 'feature',
      render: (text) => formatFeatureName(text),
    },
    {
      title: '特征值',
      dataIndex: 'value',
      key: 'value',
      render: (text, record) => formatFeatureValue(record.feature, text),
    },
    {
      title: '重要性',
      dataIndex: 'importance',
      key: 'importance',
      sorter: (a, b) => a.importance - b.importance,
      render: (value) => `${(value * 100).toFixed(1)}%`,
    },
    {
      title: '影响',
      dataIndex: 'effect',
      key: 'effect',
      render: (text) => (
        text === 'positive' ? 
          <Tag color="green" icon={<RiseOutlined />}>提升价格</Tag> : 
          <Tag color="red" icon={<FallOutlined />}>降低价格</Tag>
      ),
    },
  ];
  
  // 格式化特征名称
  const formatFeatureName = (name) => {
    const nameMap = {
      'prop_area': '建筑面积',
      'land_size': '土地面积',
      'prop_bed': '卧室数量',
      'prop_bath': '浴室数量',
      'prop_carpark': '车位数量',
      'prop_type': '房产类型',
      'locality_name': '区域名称',
      'prop_build_year': '建造年份',
      'prop_floor': '所在楼层',
      'prop_x': '经度',
      'prop_y': '纬度',
      // 添加更多特征名称映射...
    };
    
    return nameMap[name] || name;
  };
  
  // 格式化特征值
  const formatFeatureValue = (feature, value) => {
    if (feature === 'prop_area' || feature === 'land_size') return `${value}m²`;
    if (feature === 'prop_build_year') return `${value}年`;
    if (feature === 'prop_carpark') return `${value}个`;
    if (feature === 'locality_name') return value;
    if (feature === 'prop_floor') return `${value}层`;
    if (feature === 'prop_x' || feature === 'prop_y') return value.toFixed(6);
    
    return value;
  };

  // 可比房产表格列
  const comparableColumns = [
    {
      title: '地址',
      dataIndex: 'address',
      key: 'address',
      ellipsis: true,
    },
    {
      title: '面积',
      dataIndex: 'area',
      key: 'area',
      render: (text) => text ? `${text}m²` : '未知',
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      render: (text) => text ? `A$${(text / 10000).toFixed(0)}万` : '未知',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (text) => (
        <Tag color={
          text && text.includes('公寓') ? 'blue' : 
          text && text.includes('别墅') ? 'green' : 'purple'
        }>
          {text || '未知'}
        </Tag>
      ),
    },
    {
      title: '相似度',
      dataIndex: 'similarity',
      key: 'similarity',
      render: (value) => (
        <Progress 
          percent={value} 
          size="small"
          format={(percent) => `${percent}%`}
          status="active"
        />
      ),
    },
  ];

  const exportToPdf = () => {
    if (!id) return;
    
    // 显示加载通知并获取通知关闭方法
    const notificationKey = 'pdfLoading';
    notification.open({
      key: notificationKey,
      message: '正在生成PDF报告',
      description: '请稍等，正在为您生成报告...',
      duration: 0,
      icon: <ClockCircleOutlined style={{ color: '#108ee9' }} />,
    });
    
    // 设置下载链接 - 使用简化版端点
    const downloadLink = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/properties/${id}/pdf-simple`;
    
    // 打印调试信息
    console.log('正在请求PDF下载链接:', downloadLink);
    
    // 使用fetch API下载PDF
    fetch(downloadLink)
      .then(response => {
        console.log('PDF API响应状态:', response.status);
        
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('未找到该房产数据');
          } else if (response.status === 500) {
            throw new Error('服务器生成PDF时遇到错误');
          } else {
            throw new Error(`PDF生成失败 (${response.status})`);
          }
        }
        return response.blob();
      })
      .then(blob => {
        console.log('成功获取PDF，大小:', blob.size, '字节');
        
        // 创建一个下载链接
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `房产估价报告_${id}.pdf`);
        
        // 添加到DOM并触发点击
        document.body.appendChild(link);
        link.click();
        
        // 清理
        setTimeout(() => {
          window.URL.revokeObjectURL(url);
          document.body.removeChild(link);
        }, 100);
        
        // 显示成功通知
        notification.destroy(notificationKey);
        notification.success({
          message: 'PDF生成完成',
          description: '报告已成功下载',
          duration: 3,
        });
      })
      .catch(error => {
        console.error('下载PDF出错:', error);
        
        // 显示错误通知
        notification.destroy(notificationKey);
        notification.error({
          message: 'PDF生成失败',
          description: error.message || '无法生成PDF报告，请稍后重试',
          duration: 4,
        });
      });
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
            background: '#fff', 
            padding: 24, 
            minHeight: 'calc(100vh - 64px - 69px)', 
            borderRadius: '4px', 
            marginTop: '20px' 
          }}>
            <Result
              status="404"
              title="未找到房产估价报告"
              subTitle="抱歉，您查找的房产估价报告不存在或已被删除。"
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
        <title>{property.address || '房产估价报告'} - PropertyWize</title>
        <meta name="description" content={`${property.address || '房产'}的详细估价报告和分析`} />
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
            <Breadcrumb.Item>{property.id}</Breadcrumb.Item>
          </Breadcrumb>

          {/* 页面标题 */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <Title level={2}>{property.address}</Title>
            <Space>
              <Button type="primary" icon={<ExportOutlined />} onClick={exportToPdf}>导出PDF报告</Button>
              <Button type="default" icon={<ArrowLeftOutlined />}>
                <Link href="/property-reports">返回列表</Link>
              </Button>
            </Space>
          </div>

          {/* 使用Tabs组件在同一页面切换内容 */}
          <Tabs 
            activeKey={activeTab} 
            onChange={handleTabChange}
            items={[
              {
                key: '1',
                label: (
                  <span>
                    <DollarOutlined />
                    估价分析
                  </span>
                ),
                children: (
                  <Row>
                    <Col xs={24}>
                      <PropertyPriceResult property={property} />
                      <PropertyFeatureInfluences featureImportance={property.feature_importance} />
                      <PropertyComparables comparableProperties={property.comparable_properties} />
                      <PropertyConfidence predictedPrice={property.predicted_price} />
                    </Col>
                  </Row>
                )
              },
              {
                key: '2',
                label: (
                  <span>
                    <InfoCircleOutlined />
                    房产基础信息
                  </span>
                ),
                children: (
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
                )
              }
            ]}
          />
        </div>
      </Content>

      <Footer style={{ textAlign: 'center', marginTop: '24px' }}>
        PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
      </Footer>
    </Layout>
  );
} 