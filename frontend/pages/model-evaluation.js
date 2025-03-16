import { useState, useEffect } from 'react';
import Head from 'next/head';
import { 
  Layout, 
  Typography, 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Table, 
  Divider,
  Spin,
  Alert,
  Progress,
  Tooltip,
  Tag,
  Button
} from 'antd';
import { 
  BarChartOutlined,
  FileTextOutlined,
  LineChartOutlined,
  PercentageOutlined,
  SettingOutlined
} from '@ant-design/icons';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import MainHeader from '../components/Header';

// 使用动态导入并禁用 SSR
const Column = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Column),
  { ssr: false, loading: () => <div style={{ height: '400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Content, Footer } = Layout;
const { Title, Paragraph } = Typography;

export default function ModelEvaluation() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchModelInfo() {
      try {
        setLoading(true);
        const response = await fetch('/api/model/info');
        if (!response.ok) {
          throw new Error('获取模型信息失败');
        }
        const data = await response.json();
        setModelInfo(data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    }

    fetchModelInfo();
  }, []);

  // 特征重要性表格列定义
  const featureColumns = [
    {
      title: '特征名称',
      dataIndex: 'feature',
      key: 'feature',
      render: (text) => {
        // 特征名称翻译
        const translations = {
          'prop_area': '建筑面积',
          'prop_bed': '卧室数量',
          'prop_bath': '浴室数量',
          'prop_age': '房屋年龄',
          'land_size': '土地面积',
          'garage_spaces': '车库数量',
          'num_schools': '学校数量',
          'distance_cbd': '距市中心',
          'distance_train': '距火车站',
          'distance_beach': '距海滩',
        };
        
        return translations[text] || text;
      }
    },
    {
      title: '重要性得分',
      dataIndex: 'importance',
      key: 'importance',
      sorter: (a, b) => a.importance - b.importance,
      render: (value) => value.toFixed(4),
    },
    {
      title: '影响方向',
      dataIndex: 'direction',
      key: 'direction',
      render: (direction) => {
        const color = direction === 'positive' ? 'green' : 'red';
        const text = direction === 'positive' ? '正向影响' : '负向影响';
        return <Tag color={color}>{text}</Tag>;
      }
    },
    {
      title: '重要性占比',
      dataIndex: 'importance',
      key: 'importance_percent',
      render: (value, record, index) => {
        // 计算最大值用于百分比显示
        const maxImportance = Math.max(...getFeatureImportanceData().map(item => item.importance));
        const percent = (value / maxImportance) * 100;
        
        return (
          <Progress 
            percent={percent} 
            size="small"
            format={(percent) => `${percent.toFixed(1)}%`}
            status={record.direction === 'positive' ? 'success' : 'exception'}
            strokeColor={record.direction === 'positive' ? '#52c41a' : '#f5222d'}
          />
        );
      },
    },
  ];

  // 预处理特征重要性数据
  const getFeatureImportanceData = () => {
    if (!modelInfo || !modelInfo.metrics || !modelInfo.metrics.feature_importance) {
      return [];
    }
    
    // 添加方向信息和影响判断
    return modelInfo.metrics.feature_importance.map((item, index) => {
      // 简单逻辑判断影响方向 - 可以根据实际情况调整
      let direction = 'positive';
      
      if (item.feature === 'prop_age' || 
          item.feature.includes('distance_') || 
          item.feature.includes('older_than_')) {
        direction = 'negative';
      }
      
      return {
        key: index,
        feature: item.feature,
        importance: item.importance,
        direction: direction,
      };
    });
  };

  // 获取误差分布数据
  const getErrorDistribution = () => {
    if (!modelInfo || !modelInfo.metrics || !modelInfo.metrics.error_distribution) {
      return null;
    }
    
    return modelInfo.metrics.error_distribution;
  };
  
  // 获取百分位数据
  const getErrorPercentiles = () => {
    const distribution = getErrorDistribution();
    if (!distribution || !distribution.percentiles) {
      return null;
    }
    
    return distribution.percentiles;
  };
  
  // 获取误差范围分布数据
  const getErrorRanges = () => {
    const distribution = getErrorDistribution();
    if (!distribution || !distribution.error_ranges) {
      return null;
    }
    
    return distribution.error_ranges;
  };

  return (
    <Layout className="layout">
      <Head>
        <title>模型评估 - PropertyWize</title>
        <meta name="description" content="XGBoost模型性能评估和特征重要性分析" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <MainHeader selectedKey="2" />

      <Content style={{ padding: '0 50px', marginTop: 64 }}>
        <div style={{ background: '#fff', padding: 24, minHeight: 'calc(100vh - 64px - 69px)', borderRadius: '4px', marginTop: '20px' }}>
          <Title level={2}>XGBoost模型评估</Title>
          <Row justify="space-between">
            <Col>
              <Paragraph>
                本页面展示了我们XGBoost房价预测模型的详细评估指标和特征重要性分析，帮助您了解模型的性能和预测依据。
              </Paragraph>
            </Col>
            <Col>
              <Button type="primary" icon={<SettingOutlined />}>
                <Link href="/model-management" style={{ color: 'white' }}>
                  模型管理
                </Link>
              </Button>
            </Col>
          </Row>

          {loading ? (
            <div style={{ textAlign: 'center', margin: '50px 0' }}>
              <Spin size="large" />
              <p style={{ marginTop: 20 }}>加载模型评估数据...</p>
            </div>
          ) : error ? (
            <Alert
              message="加载错误"
              description={error}
              type="error"
              showIcon
            />
          ) : (
            <>
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="模型类型"
                      value={modelInfo?.model_type || 'XGBRegressor'}
                      valueStyle={{ color: '#1890ff' }}
                      prefix={<BarChartOutlined />}
                    />
                  </Card>
                </Col>

                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="特征数量"
                      value={modelInfo?.features_count || 0}
                      valueStyle={{ color: '#3f8600' }}
                    />
                  </Card>
                </Col>

                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="RMSE"
                      value={modelInfo?.metrics?.rmse?.toFixed(2) || '-'}
                      precision={2}
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Card>
                </Col>

                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="R²得分"
                      value={modelInfo?.metrics?.r2_score?.toFixed(4) || '-'}
                      precision={4}
                      valueStyle={{ color: '#1890ff' }}
                      suffix=""
                    />
                  </Card>
                </Col>
              </Row>

              <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="中位百分比误差"
                      value={modelInfo?.metrics?.median_percentage_error?.toFixed(2) || '-'}
                      precision={2}
                      valueStyle={{ color: '#722ed1' }}
                      prefix={<PercentageOutlined />}
                    />
                  </Card>
                </Col>
                
                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="平均百分比误差"
                      value={modelInfo?.metrics?.mean_percentage_error?.toFixed(2) || '-'}
                      precision={2}
                      valueStyle={{ color: '#eb2f96' }}
                      prefix={<PercentageOutlined />}
                    />
                  </Card>
                </Col>
              </Row>

              <Divider />

              <Title level={3}>误差分布分析</Title>
              <Paragraph>
                下面展示了预测误差的详细分布信息，帮助您了解模型的预测准确度分布情况。
              </Paragraph>

              {getErrorPercentiles() && (
                <>
                  <Title level={4}>误差百分位</Title>
                  <Row gutter={[16, 16]}>
                    <Col xs={24} sm={12} md={8} lg={4}>
                      <Card>
                        <Statistic
                          title="10%分位"
                          value={getErrorPercentiles().p10?.toFixed(2) || '-'}
                          precision={2}
                          suffix="%"
                        />
                      </Card>
                    </Col>
                    <Col xs={24} sm={12} md={8} lg={4}>
                      <Card>
                        <Statistic
                          title="25%分位"
                          value={getErrorPercentiles().p25?.toFixed(2) || '-'}
                          precision={2}
                          suffix="%"
                        />
                      </Card>
                    </Col>
                    <Col xs={24} sm={12} md={8} lg={4}>
                      <Card>
                        <Statistic
                          title="中位数(50%)"
                          value={getErrorPercentiles().p50?.toFixed(2) || '-'}
                          precision={2}
                          suffix="%"
                          valueStyle={{ color: '#1890ff' }}
                        />
                      </Card>
                    </Col>
                    <Col xs={24} sm={12} md={8} lg={4}>
                      <Card>
                        <Statistic
                          title="75%分位"
                          value={getErrorPercentiles().p75?.toFixed(2) || '-'}
                          precision={2}
                          suffix="%"
                        />
                      </Card>
                    </Col>
                    <Col xs={24} sm={12} md={8} lg={4}>
                      <Card>
                        <Statistic
                          title="90%分位"
                          value={getErrorPercentiles().p90?.toFixed(2) || '-'}
                          precision={2}
                          suffix="%"
                        />
                      </Card>
                    </Col>
                  </Row>
                </>
              )}

              {getErrorRanges() && (
                <>
                  <Title level={4} style={{ marginTop: '24px' }}>误差范围分布</Title>
                  <Row gutter={[16, 16]}>
                    <Col span={24}>
                      <Card>
                        <Row>
                          <Col span={24}>
                            <Tooltip title={`误差小于5%的样本比例: ${(getErrorRanges()['<5%'] * 100).toFixed(2)}%`}>
                              <div style={{ marginBottom: '10px' }}>
                                <span>误差 &lt;5%:</span>
                                <Progress 
                                  percent={(getErrorRanges()['<5%'] * 100).toFixed(2)} 
                                  status="success" 
                                  strokeColor="#52c41a" 
                                />
                              </div>
                            </Tooltip>
                            
                            <Tooltip title={`误差在5-10%之间的样本比例: ${(getErrorRanges()['5-10%'] * 100).toFixed(2)}%`}>
                              <div style={{ marginBottom: '10px' }}>
                                <span>误差 5-10%:</span>
                                <Progress 
                                  percent={(getErrorRanges()['5-10%'] * 100).toFixed(2)} 
                                  status="active" 
                                  strokeColor="#1890ff" 
                                />
                              </div>
                            </Tooltip>
                            
                            <Tooltip title={`误差在10-15%之间的样本比例: ${(getErrorRanges()['10-15%'] * 100).toFixed(2)}%`}>
                              <div style={{ marginBottom: '10px' }}>
                                <span>误差 10-15%:</span>
                                <Progress 
                                  percent={(getErrorRanges()['10-15%'] * 100).toFixed(2)} 
                                  status="active" 
                                  strokeColor="#faad14" 
                                />
                              </div>
                            </Tooltip>
                            
                            <Tooltip title={`误差在15-20%之间的样本比例: ${(getErrorRanges()['15-20%'] * 100).toFixed(2)}%`}>
                              <div style={{ marginBottom: '10px' }}>
                                <span>误差 15-20%:</span>
                                <Progress 
                                  percent={(getErrorRanges()['15-20%'] * 100).toFixed(2)} 
                                  status="active" 
                                  strokeColor="#fa8c16" 
                                />
                              </div>
                            </Tooltip>
                            
                            <Tooltip title={`误差大于20%的样本比例: ${(getErrorRanges()['>20%'] * 100).toFixed(2)}%`}>
                              <div>
                                <span>误差 &gt;20%:</span>
                                <Progress 
                                  percent={(getErrorRanges()['>20%'] * 100).toFixed(2)} 
                                  status="exception" 
                                  strokeColor="#f5222d" 
                                />
                              </div>
                            </Tooltip>
                          </Col>
                        </Row>
                      </Card>
                    </Col>
                  </Row>
                </>
              )}

              <Divider />

              <Title level={3}>特征重要性</Title>
              <Paragraph>
                下表展示了模型中最重要的特征及其对预测结果的影响程度。特征重要性分数表示每个变量在模型预测中的相对重要性。
              </Paragraph>

              <Row gutter={[16, 16]}>
                <Col xs={24} lg={16}>
                  <Table 
                    columns={featureColumns} 
                    dataSource={getFeatureImportanceData()}
                    pagination={{ pageSize: 10 }}
                    rowClassName={(record, index) => index % 2 === 0 ? 'table-row-light' : 'table-row-dark'}
                  />
                </Col>
                
                <Col xs={24} lg={8}>
                  <Card title="特征重要性可视化">
                    <Paragraph>
                      下图展示了模型中最重要的10个特征，按重要性降序排列：
                    </Paragraph>
                    {modelInfo?.metrics?.feature_importance && (
                      <Column 
                        data={getFeatureImportanceData().slice(0, 10)}
                        xField="importance"
                        yField="feature"
                        seriesField="direction"
                        color={({ direction }) => {
                          return direction === 'positive' ? '#52c41a' : '#f5222d';
                        }}
                        label={{
                          position: 'right',
                          formatter: (datum) => `${datum.importance.toFixed(3)}`,
                        }}
                        xAxis={{
                          title: {
                            text: '重要性分数',
                          },
                        }}
                        yAxis={{
                          title: {
                            text: '特征',
                          },
                        }}
                      />
                    )}
                  </Card>
                </Col>
              </Row>

              <Divider />
              
              <Title level={3}>预测性能</Title>
              <Paragraph>
                模型在测试集上的预测效果指标，用于评估模型泛化能力。
              </Paragraph>

              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="平均绝对误差 (MAE)"
                      value={modelInfo?.metrics?.mae?.toFixed(2) || '-'}
                      precision={2}
                    />
                  </Card>
                </Col>
                
                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="均方误差 (MSE)"
                      value={modelInfo?.metrics?.mse?.toFixed(2) || '-'}
                      precision={2}
                    />
                  </Card>
                </Col>
                
                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="中位数绝对误差 (MedianAE)"
                      value={modelInfo?.metrics?.median_ae?.toFixed(2) || '-'}
                      precision={2}
                    />
                  </Card>
                </Col>
                
                <Col xs={24} sm={12} md={8} lg={6}>
                  <Card>
                    <Statistic
                      title="解释方差比"
                      value={modelInfo?.metrics?.explained_variance?.toFixed(4) || '-'}
                      precision={4}
                    />
                  </Card>
                </Col>
              </Row>
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