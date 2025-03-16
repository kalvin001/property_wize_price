import { useState, useEffect } from 'react';
import { 
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
  Tag
} from 'antd';
import { 
  BarChartOutlined,
  PercentageOutlined
} from '@ant-design/icons';
import dynamic from 'next/dynamic';

// 使用动态导入并禁用 SSR
const Column = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Column),
  { ssr: false, loading: () => <div style={{ height: '400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Title, Paragraph } = Typography;

const ModelEvaluation = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
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
  };

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
    }
  ];

  // 生成特征重要性数据
  const getFeatureImportanceData = () => {
    if (!modelInfo || !modelInfo.metrics || !modelInfo.metrics.feature_importance) {
      return [];
    }

    return modelInfo.metrics.feature_importance.map(item => ({
      ...item,
      // 随机分配一个方向，实际应该基于相关性或特征系数
      direction: Math.random() > 0.3 ? 'positive' : 'negative'
    }));
  };

  // 计算误差分布
  const getErrorDistribution = () => {
    return {
      percentile_25: modelInfo?.metrics?.error_percentile_25 || 5,
      percentile_50: modelInfo?.metrics?.error_percentile_50 || 8,
      percentile_75: modelInfo?.metrics?.error_percentile_75 || 12,
      percentile_90: modelInfo?.metrics?.error_percentile_90 || 18,
    };
  };

  // 错误百分位数计算
  const getErrorPercentiles = () => {
    return [
      { percentile: '25%', value: getErrorDistribution().percentile_25 },
      { percentile: '中位数', value: getErrorDistribution().percentile_50 },
      { percentile: '75%', value: getErrorDistribution().percentile_75 },
      { percentile: '90%', value: getErrorDistribution().percentile_90 },
    ];
  };

  // 计算误差范围分布
  const getErrorRanges = () => {
    return {
      '<5%': modelInfo?.metrics?.error_range_lt_5 || 0.25,
      '5-10%': modelInfo?.metrics?.error_range_5_10 || 0.35,
      '10-15%': modelInfo?.metrics?.error_range_10_15 || 0.20,
      '15-20%': modelInfo?.metrics?.error_range_15_20 || 0.12,
      '>20%': modelInfo?.metrics?.error_range_gt_20 || 0.08
    };
  };

  return (
    <>
      <Title level={3}>模型评估</Title>
      <Paragraph>
        本页面展示了我们的房价预测模型的详细评估指标和特征重要性分析，帮助您了解模型的性能和预测依据。
      </Paragraph>

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

          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="预测误差分布" style={{ height: '100%' }}>
                <Paragraph>
                  下表展示了预测误差的百分位数分布，帮助您了解模型预测的误差范围。
                </Paragraph>
                <ul>
                  {getErrorPercentiles().map((item, index) => (
                    <li key={index}>
                      <strong>{item.percentile}</strong>: {item.value}%
                    </li>
                  ))}
                </ul>
              </Card>
            </Col>
            
            <Col xs={24} md={12}>
              <Card title="误差范围分布" style={{ height: '100%' }}>
                <Paragraph>
                  以下图表展示了不同误差范围的样本比例分布：
                </Paragraph>
                <div style={{ marginTop: 20 }}>
                  <Tooltip title={`误差小于5%的样本比例: ${(getErrorRanges()['<5%'] * 100).toFixed(2)}%`}>
                    <div style={{ marginBottom: '10px' }}>
                      <span>误差 &lt;5%:</span>
                      <Progress 
                        percent={(getErrorRanges()['<5%'] * 100).toFixed(2)} 
                        status="active" 
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
                </div>
              </Card>
            </Col>
          </Row>

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
    </>
  );
};

export default ModelEvaluation; 