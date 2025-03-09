import React, { useEffect, useState } from 'react';
import { Layout, Typography, Card, Spin, Tabs, Table, Statistic, Row, Col, Divider, Tag, Progress } from 'antd';
import './App.css';

const { Header, Content, Footer } = Layout;
const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

// 定义数据接口
interface ModelInfo {
  model_type: string;
  features_count: number;
  feature_names: string[];
  metrics: {
    xgboost: {
      rmse: number;
      mae: number;
      r2: number;
    }
  };
  data_info: {
    total_records: number;
    features_count: number;
    price_range: {
      min: number;
      max: number;
      mean: number;
      median: number;
    }
  }
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface PropertyFeature {
  feature: string;
  value: number;
  shap_value: number;
  is_positive: boolean;
}

interface SampleProperty {
  prop_id: string;
  address: string;
  predicted_price: number;
  actual_price: number;
  error_percent: number;
  top_features: PropertyFeature[];
}

const App: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [sampleProperties, setSampleProperties] = useState<SampleProperty[]>([]);
  const [activeTab, setActiveTab] = useState<string>('overview');

  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取模型信息
        const modelInfoResponse = await fetch('/api/model/info');
        if (modelInfoResponse.ok) {
          const modelInfoData = await modelInfoResponse.json();
          setModelInfo(modelInfoData);
        } else {
          console.error('无法获取模型信息:', modelInfoResponse.statusText);
        }

        // 获取特征重要性 - 使用相对路径
        try {
          const featureImportanceResponse = await fetch('/data/feature_importance.json');
          if (featureImportanceResponse.ok) {
            const featureImportanceData = await featureImportanceResponse.json();
            setFeatureImportance(featureImportanceData);
          } else {
            console.error('无法获取特征重要性数据:', featureImportanceResponse.statusText);
          }
        } catch (error) {
          console.error('获取特征重要性时出错:', error);
        }

        // 获取样本房产
        const samplePropertiesResponse = await fetch('/api/properties/sample');
        if (samplePropertiesResponse.ok) {
          const samplePropertiesData = await samplePropertiesResponse.json();
          setSampleProperties(samplePropertiesData);
        } else {
          console.error('无法获取样本房产数据:', samplePropertiesResponse.statusText);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // 格式化价格
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('zh-CN', {
      style: 'currency',
      currency: 'CNY',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(price);
  };

  // 渲染总览
  const renderOverview = () => {
    if (!modelInfo) return null;

    return (
      <div className="overview-container">
        <Card title="模型概述" className="overview-card">
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Statistic title="模型类型" value={modelInfo.model_type} />
            </Col>
            <Col span={8}>
              <Statistic title="特征数量" value={modelInfo.features_count} />
            </Col>
            <Col span={8}>
              <Statistic 
                title="模型精确度 (R²)" 
                value={modelInfo.metrics?.xgboost?.r2 || 0} 
                precision={3}
                suffix="" 
              />
            </Col>
          </Row>
          <Divider />
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Statistic 
                title="均价" 
                value={modelInfo.data_info?.price_range?.mean || 0} 
                precision={0}
                formatter={(value) => formatPrice(Number(value))}
              />
            </Col>
            <Col span={8}>
              <Statistic 
                title="最低价" 
                value={modelInfo.data_info?.price_range?.min || 0} 
                precision={0}
                formatter={(value) => formatPrice(Number(value))}
              />
            </Col>
            <Col span={8}>
              <Statistic 
                title="最高价" 
                value={modelInfo.data_info?.price_range?.max || 0} 
                precision={0}
                formatter={(value) => formatPrice(Number(value))}
              />
            </Col>
          </Row>
        </Card>

        <Card title="模型评估指标" className="metrics-card" style={{ marginTop: 16 }}>
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Statistic 
                title="决定系数 (R²)" 
                value={modelInfo.metrics?.xgboost?.r2 || 0} 
                precision={3}
                valueStyle={{ color: '#3f8600' }}
              />
              <Paragraph className="metric-description">
                衡量模型解释变异性的能力，值越接近1表示模型越好。
              </Paragraph>
            </Col>
            <Col span={8}>
              <Statistic 
                title="均方根误差 (RMSE)" 
                value={modelInfo.metrics?.xgboost?.rmse || 0} 
                precision={0}
                formatter={(value) => formatPrice(Number(value))}
                valueStyle={{ color: '#cf1322' }}
              />
              <Paragraph className="metric-description">
                预测值与实际值之间的平均误差，单位与房价相同。
              </Paragraph>
            </Col>
            <Col span={8}>
              <Statistic 
                title="平均绝对误差 (MAE)" 
                value={modelInfo.metrics?.xgboost?.mae || 0} 
                precision={0}
                formatter={(value) => formatPrice(Number(value))}
                valueStyle={{ color: '#cf1322' }}
              />
              <Paragraph className="metric-description">
                预测值与实际值的平均绝对差，单位与房价相同。
              </Paragraph>
            </Col>
          </Row>
        </Card>
      </div>
    );
  };

  // 渲染特征重要性
  const renderFeatureImportance = () => {
    const columns = [
      {
        title: '排名',
        dataIndex: 'rank',
        key: 'rank',
        width: 80,
      },
      {
        title: '特征名称',
        dataIndex: 'feature',
        key: 'feature',
      },
      {
        title: '重要性得分',
        dataIndex: 'importance',
        key: 'importance',
        render: (importance: number) => (
          <Progress 
            percent={Math.round(importance * 100 / (featureImportance[0]?.importance || 1))} 
            format={(percent) => `${importance.toFixed(4)}`}
            status="active"
          />
        ),
      },
    ];

    const data = featureImportance.map((item, index) => ({
      key: index,
      rank: index + 1,
      feature: item.feature,
      importance: item.importance,
    }));

    return (
      <div className="feature-importance-container">
        <Card title="特征重要性分析" className="feature-importance-card">
          <Paragraph>
            下表展示了影响房价的最重要特征，由随机森林模型计算得出。这些特征对房价预测有显著影响。
          </Paragraph>
          <Table 
            columns={columns} 
            dataSource={data} 
            pagination={false}
            className="feature-table"
          />
        </Card>
      </div>
    );
  };

  // 渲染样本房产
  const renderSampleProperties = () => {
    return (
      <div className="sample-properties-container">
        {sampleProperties.map((property, index) => (
          <Card 
            key={property.prop_id} 
            title={`样本房产 #${index + 1}: ${property.address}`} 
            className="property-card"
            style={{ marginBottom: 16 }}
          >
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Statistic 
                  title="实际价格" 
                  value={property.actual_price} 
                  precision={0}
                  formatter={(value) => formatPrice(Number(value))}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="预测价格" 
                  value={property.predicted_price} 
                  precision={0}
                  formatter={(value) => formatPrice(Number(value))}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="误差百分比" 
                  value={property.error_percent} 
                  precision={2}
                  suffix="%"
                  valueStyle={{ 
                    color: Math.abs(property.error_percent) < 10 ? '#3f8600' : 
                           Math.abs(property.error_percent) < 20 ? '#faad14' : '#cf1322' 
                  }}
                />
              </Col>
            </Row>
            <Divider orientation="left">主要影响因素</Divider>
            <div className="feature-impact">
              {property.top_features.map((feature, featureIndex) => (
                <div key={featureIndex} className="feature-item">
                  <Tag color={feature.is_positive ? 'green' : 'red'}>
                    {feature.is_positive ? '↑' : '↓'} {feature.feature}
                  </Tag>
                  <Text>
                    值: {feature.value.toFixed(2)} 
                    {feature.is_positive ? ' 提高' : ' 降低'}了价格 
                    约 {formatPrice(Math.abs(feature.shap_value))}
                  </Text>
                </div>
              ))}
            </div>
            <Divider />
            <Paragraph className="property-explanation">
              <Text strong>解释: </Text>
              该房产的预测价格与实际价格相差{Math.abs(property.error_percent).toFixed(2)}%，
              {property.error_percent > 0 ? '高于' : '低于'}实际值。
              {property.top_features[0]?.feature && <>主要由 <Text mark>{property.top_features[0].feature}</Text> 特征影响</>}。
            </Paragraph>
          </Card>
        ))}
      </div>
    );
  };

  return (
    <Layout className="layout">
      <Header className="header">
        <div className="logo" />
        <Title level={3} className="site-title">房产估价分析系统</Title>
      </Header>
      <Content className="content">
        {loading ? (
          <div className="loading-container">
            <Spin size="large" />
            <p>加载数据中...</p>
          </div>
        ) : (
          <div className="content-container">
            <Tabs activeKey={activeTab} onChange={setActiveTab} className="main-tabs">
              <TabPane tab="模型概览" key="overview">
                {renderOverview()}
              </TabPane>
              <TabPane tab="特征重要性" key="features">
                {renderFeatureImportance()}
              </TabPane>
              <TabPane tab="样本房产分析" key="samples">
                {renderSampleProperties()}
              </TabPane>
            </Tabs>
          </div>
        )}
      </Content>
      <Footer className="footer">房产估价分析系统 ©{new Date().getFullYear()} 基于XGBoost算法的可解释性分析</Footer>
    </Layout>
  );
};

export default App; 