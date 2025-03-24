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
  Button,
  Tabs,
  Empty,
  List,
  Badge
} from 'antd';
import { 
  BarChartOutlined,
  FileTextOutlined,
  LineChartOutlined,
  PercentageOutlined,
  SettingOutlined,
  ArrowLeftOutlined,
  CheckCircleOutlined,
  SyncOutlined
} from '@ant-design/icons';
import dynamic from 'next/dynamic';
import MainHeader from '../components/Header';

// 使用动态导入并禁用 SSR
const Column = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Column),
  { ssr: false, loading: () => <div style={{ height: '400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

export default function ModelCenter() {
  const [modelInfo, setModelInfo] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('1');
  const [selectedModel, setSelectedModel] = useState(null);
  const [showModelDetail, setShowModelDetail] = useState(false);
  const [activeModel, setActiveModel] = useState(null);

  // 获取模型列表和当前激活的模型
  useEffect(() => {
    async function fetchModels() {
      try {
        setLoading(true);
        
        // 获取模型列表
        const modelsResponse = await fetch('/api/models');
        console.log("modelsResponse===", modelsResponse);
        if (!modelsResponse.ok) {
          throw new Error('获取模型列表失败');
        }
        const modelsData = await modelsResponse.json();
        console.log("模型列表数据:", modelsData.models);
        modelsData.models.forEach(model => {
          console.log(`模型 ${model.name} 指标:`, model.metrics);
          console.log(`  - 中位误差: ${model.metrics?.median_percentage_error}`);
          console.log(`  - 误差<10%比例: ${model.metrics?.error_under_10_percent}`);
        });
        setModels(modelsData.models || []);
        
        // 获取当前激活的模型信息
        const infoResponse = await fetch('/api/model/info');
        if (infoResponse.ok) {
          const infoData = await infoResponse.json();
          setModelInfo(infoData);
          if (infoData.model_path) {
            setActiveModel(infoData.model_path);
          }
        }
        
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    }

    fetchModels();
  }, []);

  // 选择模型查看详情
  const handleViewModelDetail = (model) => {
    setSelectedModel(model);
    setShowModelDetail(true);
    
    // 获取模型详情
    fetchModelDetail(model.path);
  };

  // 返回模型列表
  const handleBackToList = () => {
    setShowModelDetail(false);
    setSelectedModel(null);
  };

  // 获取模型详细信息
  const fetchModelDetail = async (modelPath) => {
    try {
      setLoading(true);
      
      // 从模型路径提取模型名称
      const modelName = modelPath.replace('../model/', '').replace('_model.joblib', '');
      console.log(`获取模型详情，模型名称: ${modelName}`);
      
      // 直接访问metrics文件，注意路径前缀调整为后端访问路径
      const response = await fetch(`/api/models/${encodeURIComponent(modelName)}/metrics`);
      if (!response.ok) {
        throw new Error(`获取模型详情失败，状态码: ${response.status}`);
      }
      const data = await response.json();
      console.log("模型详情数据:", data);
      
      // 数据有效性验证
      if (!data) {
        console.error("接收到空的模型详情数据");
        setError("接收到空的模型详情数据");
        setLoading(false);
        return;
      }
      
      // 检查关键指标
      console.log("指标数据验证:");
      console.log("- RMSE:", data.rmse);
      console.log("- R2 Score:", data.r2_score);
      console.log("- 中位误差:", data.median_percentage_error);
      console.log("- 误差<10%比例:", data.error_under_10_percent);
      console.log("- 特征重要性:", data.feature_importance ? `${data.feature_importance.length} 项` : "无");
      console.log("- 误差分布:", data.error_distribution ? "存在" : "无");
      
      // 设置数据
      setModelInfo(data);
      setLoading(false);
    } catch (err) {
      console.error("获取模型详情失败:", err);
      setError(err.message);
      setLoading(false);
    }
  };

  // 预处理特征重要性数据
  const getFeatureImportanceData = () => {
    if (!modelInfo || !modelInfo.feature_importance) {
      console.log("没有特征重要性数据");
      return [];
    }
    
    console.log("处理特征重要性数据:", modelInfo.feature_importance);
    
    // 添加方向信息和影响判断
    return modelInfo.feature_importance.map((item, index) => {
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
    if (!modelInfo || !modelInfo.error_distribution) {
      console.log("没有误差分布数据");
      return null;
    }
    
    console.log("误差分布数据:", modelInfo.error_distribution);
    return modelInfo.error_distribution;
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

  // 模型列表渲染
  const renderModelList = () => {
    return (
      <>
        <div style={{ marginBottom: 16 }}>
          <Title level={2}>模型列表</Title>
          <Paragraph>
            下面展示了所有可用的房价预测模型，您可以查看它们的基本性能指标，并点击卡片查看详细信息。
          </Paragraph>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', margin: '50px 0' }}>
            <Spin size="large" />
            <p style={{ marginTop: 20 }}>加载模型列表...</p>
          </div>
        ) : error ? (
          <Alert
            message="加载错误"
            description={error}
            type="error"
            showIcon
          />
        ) : models.length === 0 ? (
          <Empty description="暂无模型" />
        ) : (
          <List
            grid={{ gutter: 16, xs: 1, sm: 1, md: 2, lg: 3, xl: 3, xxl: 4 }}
            dataSource={models}
            renderItem={(model) => (
              <List.Item>
                <Card 
                  title={
                    <div>
                      {model.name}
                      {activeModel === model.path && (
                        <Badge status="processing" text="当前激活" style={{ marginLeft: 8, color: '#1890ff' }} />
                      )}
                    </div>
                  }
                  hoverable
                  onClick={() => handleViewModelDetail(model)}
                  actions={[
                    <Button type="link" onClick={(e) => {
                      e.stopPropagation();
                      handleViewModelDetail(model);
                    }}>查看详情</Button>
                  ]}
                >
                  <Statistic
                    title="模型类型"
                    value={model.type || 'Unknown'}
                    valueStyle={{ color: '#1890ff' }}
                    prefix={<BarChartOutlined />}
                  />
                  <Divider style={{ margin: '12px 0' }} />
                  <Row gutter={[8, 8]}>
                    <Col span={12}>
                      <Statistic
                        title="中位数误差"
                        value={model.metrics?.median_percentage_error || '-'}
                        precision={3}
                        valueStyle={{ fontSize: '14px' }}
                        suffix="%"
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="误差<10%比例"
                        value={model.metrics?.error_under_10_percent || '-'}
                        precision={3}
                        valueStyle={{ fontSize: '14px', color: '#52c41a' }}
                        suffix="%"
                      />
                    </Col>
                  </Row>
                  <Row style={{ marginTop: 4 }}>
                    <Col span={24}>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <Text>状态: </Text>
                        {model.status === 'active' ? (
                          <Tag color="green" style={{ marginLeft: 4 }}>
                            <CheckCircleOutlined /> 已激活
                          </Tag>
                        ) : (
                          <Tag color="default" style={{ marginLeft: 4 }}>
                            <SyncOutlined /> 可用
                          </Tag>
                        )}
                      </div>
                    </Col>
                  </Row>
                </Card>
              </List.Item>
            )}
          />
        )}
      </>
    );
  };

  // 模型详情页面
  const renderModelDetail = () => {
    if (!selectedModel) {
      return <Empty description="请选择一个模型查看详情" />;
    }
    
    return (
      <>
        <div style={{ marginBottom: 24 }}>
          <Button 
            type="link" 
            icon={<ArrowLeftOutlined />} 
            onClick={handleBackToList}
            style={{ padding: 0, marginBottom: 16 }}
          >
            返回模型列表
          </Button>
          <Title level={2}>{selectedModel.name} 模型评估</Title>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
            <Text>模型状态: </Text>
            {activeModel === selectedModel.path ? (
              <Tag color="green" style={{ marginLeft: 8 }}>
                <CheckCircleOutlined /> 当前激活
              </Tag>
            ) : (
              <Tag color="default" style={{ marginLeft: 8 }}>
                <SyncOutlined /> 未激活
              </Tag>
            )}
          </div>
          <Paragraph>
            本页面展示了 {selectedModel.name} 模型的详细评估指标和特征重要性分析，帮助您了解模型的性能和预测依据。
          </Paragraph>
        </div>

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
                    value={selectedModel.type || 'Unknown'}
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
                    value={(modelInfo?.rmse !== null && modelInfo?.rmse !== undefined) ? modelInfo.rmse.toFixed(2) : '-'}
                    precision={2}
                    valueStyle={{ color: '#cf1322' }}
                  />
                </Card>
              </Col>

              <Col xs={24} sm={12} md={8} lg={6}>
                <Card>
                  <Statistic
                    title="R²得分"
                    value={(modelInfo?.r2_score !== null && modelInfo?.r2_score !== undefined) ? modelInfo.r2_score.toFixed(4) : '-'}
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
                    value={(modelInfo?.median_percentage_error !== null && modelInfo?.median_percentage_error !== undefined) ? modelInfo.median_percentage_error : '-'}
                    precision={3}
                    valueStyle={{ color: '#722ed1' }}
                    prefix={<PercentageOutlined />}
                    suffix="%"
                  />
                </Card>
              </Col>
              
              <Col xs={24} sm={12} md={8} lg={6}>
                <Card>
                  <Statistic
                    title="误差<10%比例"
                    value={
                      getErrorRanges() ? 
                      ((getErrorRanges()['<5%'] + getErrorRanges()['5-10%']) * 100) : 
                      (modelInfo?.error_under_10_percent !== null && modelInfo?.error_under_10_percent !== undefined) ? modelInfo.error_under_10_percent : '-'
                    }
                    precision={3}
                    valueStyle={{ color: '#52c41a' }}
                    suffix="%"
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
                        value={(getErrorPercentiles().p10 !== null && getErrorPercentiles().p10 !== undefined) ? getErrorPercentiles().p10.toFixed(2) : '-'}
                        precision={2}
                        suffix="%"
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={8} lg={4}>
                    <Card>
                      <Statistic
                        title="25%分位"
                        value={(getErrorPercentiles().p25 !== null && getErrorPercentiles().p25 !== undefined) ? getErrorPercentiles().p25.toFixed(2) : '-'}
                        precision={2}
                        suffix="%"
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={8} lg={4}>
                    <Card>
                      <Statistic
                        title="中位数(50%)"
                        value={(getErrorPercentiles().p50 !== null && getErrorPercentiles().p50 !== undefined) ? getErrorPercentiles().p50.toFixed(2) : '-'}
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
                        value={(getErrorPercentiles().p75 !== null && getErrorPercentiles().p75 !== undefined) ? getErrorPercentiles().p75.toFixed(2) : '-'}
                        precision={2}
                        suffix="%"
                      />
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={8} lg={4}>
                    <Card>
                      <Statistic
                        title="90%分位"
                        value={(getErrorPercentiles().p90 !== null && getErrorPercentiles().p90 !== undefined) ? getErrorPercentiles().p90.toFixed(2) : '-'}
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
                          <div style={{ 
                            textAlign: 'center', 
                            marginBottom: '15px', 
                            fontWeight: 'bold', 
                            fontSize: '16px', 
                            color: '#52c41a' 
                          }}>
                            误差小于10%的样本比例: {
                              getErrorRanges() 
                                ? ((getErrorRanges()['<5%'] + getErrorRanges()['5-10%']) * 100).toFixed(3) 
                                : (modelInfo?.error_under_10_percent || 0).toFixed(3)
                            }%
                          </div>
                          <Tooltip title={`误差小于5%的样本比例: ${getErrorRanges() ? (getErrorRanges()['<5%'] * 100).toFixed(2) : '未知'}%`}>
                            <div style={{ marginBottom: '10px' }}>
                              <span>误差 &lt;5%:</span>
                              <Progress 
                                percent={getErrorRanges() ? (getErrorRanges()['<5%'] * 100).toFixed(2) : 0} 
                                status="success" 
                                strokeColor="#52c41a" 
                              />
                            </div>
                          </Tooltip>
                          
                          <Tooltip title={`误差在5-10%之间的样本比例: ${getErrorRanges() ? (getErrorRanges()['5-10%'] * 100).toFixed(2) : '未知'}%`}>
                            <div style={{ marginBottom: '10px' }}>
                              <span>误差 5-10%:</span>
                              <Progress 
                                percent={getErrorRanges() ? (getErrorRanges()['5-10%'] * 100).toFixed(2) : 0} 
                                status="active" 
                                strokeColor="#1890ff" 
                              />
                            </div>
                          </Tooltip>
                          
                          <Tooltip title={`误差在10-15%之间的样本比例: ${getErrorRanges() ? (getErrorRanges()['10-15%'] * 100).toFixed(2) : '未知'}%`}>
                            <div style={{ marginBottom: '10px' }}>
                              <span>误差 10-15%:</span>
                              <Progress 
                                percent={getErrorRanges() ? (getErrorRanges()['10-15%'] * 100).toFixed(2) : 0} 
                                status="active" 
                                strokeColor="#faad14" 
                              />
                            </div>
                          </Tooltip>
                          
                          <Tooltip title={`误差在15-20%之间的样本比例: ${getErrorRanges() ? (getErrorRanges()['15-20%'] * 100).toFixed(2) : '未知'}%`}>
                            <div style={{ marginBottom: '10px' }}>
                              <span>误差 15-20%:</span>
                              <Progress 
                                percent={getErrorRanges() ? (getErrorRanges()['15-20%'] * 100).toFixed(2) : 0} 
                                status="active" 
                                strokeColor="#fa8c16" 
                              />
                            </div>
                          </Tooltip>
                          
                          <Tooltip title={`误差大于20%的样本比例: ${getErrorRanges() ? (getErrorRanges()['>20%'] * 100).toFixed(2) : '未知'}%`}>
                            <div>
                              <span>误差 &gt;20%:</span>
                              <Progress 
                                percent={getErrorRanges() ? (getErrorRanges()['>20%'] * 100).toFixed(2) : 0} 
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

            {/* 如果没有误差分布数据，但有误差<10%比例，则显示简化版视图 */}
            {!getErrorRanges() && modelInfo?.error_under_10_percent && (
              <>
                <Title level={4} style={{ marginTop: '24px' }}>误差分布摘要</Title>
                <Row gutter={[16, 16]}>
                  <Col span={24}>
                    <Card>
                      <Row>
                        <Col span={24}>
                          <div style={{ 
                            textAlign: 'center', 
                            marginBottom: '15px', 
                            fontWeight: 'bold', 
                            fontSize: '16px', 
                            color: '#52c41a' 
                          }}>
                            误差小于10%的样本比例: {(modelInfo.error_under_10_percent).toFixed(3)}%
                          </div>
                          <div style={{ marginBottom: '20px' }}>
                            <span>误差 &lt;10%:</span>
                            <Progress 
                              percent={(modelInfo.error_under_10_percent)} 
                              status="success" 
                              strokeColor="#52c41a" 
                            />
                          </div>
                          <div>
                            <span>误差 &gt;10%:</span>
                            <Progress 
                              percent={(100 - modelInfo.error_under_10_percent)} 
                              status="exception" 
                              strokeColor="#f5222d" 
                            />
                          </div>
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

            {getFeatureImportanceData().length > 0 ? (
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
                    
                    {getFeatureImportanceData().length > 0 ? (
                      <Column
                        height={300}
                        data={getFeatureImportanceData().slice(0, 10).map(item => ({
                          feature: item.feature,
                          importance: item.importance,
                        }))}
                        xField="importance"
                        yField="feature"
                        seriesField="feature"
                        legend={{
                          position: 'top',
                        }}
                        label={{
                          position: 'right',
                          formatter: (datum) => `${datum.importance.toFixed(4)}`,
                        }}
                      />
                    ) : (
                      <Empty description="暂无特征重要性数据" />
                    )}
                  </Card>
                </Col>
              </Row>
            ) : (
              <Empty description="暂无特征重要性数据" />
            )}

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
                    value={(modelInfo?.mae !== null && modelInfo?.mae !== undefined) ? modelInfo.mae.toFixed(2) : '-'}
                    precision={2}
                  />
                </Card>
              </Col>
              
              <Col xs={24} sm={12} md={8} lg={6}>
                <Card>
                  <Statistic
                    title="均方误差 (MSE)"
                    value={(modelInfo?.mse !== null && modelInfo?.mse !== undefined) ? modelInfo.mse.toFixed(2) : '-'}
                    precision={2}
                  />
                </Card>
              </Col>
              
              <Col xs={24} sm={12} md={8} lg={6}>
                <Card>
                  <Statistic
                    title="中位数绝对误差 (MedianAE)"
                    value={(modelInfo?.median_ae !== null && modelInfo?.median_ae !== undefined) ? modelInfo.median_ae.toFixed(2) : '-'}
                    precision={2}
                  />
                </Card>
              </Col>
              
              <Col xs={24} sm={12} md={8} lg={6}>
                <Card>
                  <Statistic
                    title="解释方差比"
                    value={(modelInfo?.explained_variance !== null && modelInfo?.explained_variance !== undefined) ? modelInfo.explained_variance.toFixed(4) : '-'}
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

  // 模型管理内容
  const renderModelManagement = () => {
    // 注意: 这里引入的ModelManagement组件应该是不包含MainHeader和Layout的纯内容组件
    const ModelManagementContent = require('../components/ModelManagement').default;
    return <ModelManagementContent />;
  };

  return (
    <Layout className="layout">
      <Head>
        <title>模型中心 - PropertyWize</title>
        <meta name="description" content="模型中心 - 模型评估与管理" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <MainHeader selectedKey="2" />

      <Content style={{ padding: '0 50px', marginTop: 64 }}>
        <div style={{ background: '#fff', padding: 24, minHeight: 'calc(100vh - 64px - 69px)', borderRadius: '4px', marginTop: '20px' }}>
          <Tabs activeKey={activeTab} onChange={setActiveTab} type="card">
            <TabPane 
              tab={
                <span>
                  <BarChartOutlined />
                  模型列表
                </span>
              } 
              key="1"
            >
              {showModelDetail ? renderModelDetail() : renderModelList()}
            </TabPane>
            <TabPane 
              tab={
                <span>
                  <SettingOutlined />
                  模型管理
                </span>
              } 
              key="2"
            >
              {renderModelManagement()}
            </TabPane>
          </Tabs>
        </div>
      </Content>

      <Footer style={{ textAlign: 'center' }}>
        PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
      </Footer>
    </Layout>
  );
} 