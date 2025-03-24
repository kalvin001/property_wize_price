import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Typography, 
  Button, 
  Row, 
  Col, 
  Statistic, 
  Divider, 
  Tag, 
  Space, 
  Tooltip, 
  Progress,
  Modal,
  Table,
  Empty,
  Spin
} from 'antd';
import {
  CheckCircleOutlined,
  BarChartOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Text, Title, Paragraph } = Typography;

const ModelCard = ({ model, onActivate, isActive, showDetails = false }) => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  
  // 格式化数字的辅助函数
  const formatNumber = (num) => {
    if (num === null || num === undefined) return '-';
    return typeof num === 'number' ? num.toLocaleString() : num;
  };
  
  // 加载模型的详细评估指标
  const loadModelMetrics = async () => {
    if (!model) return;
    
    setLoading(true);
    
    // 提取模型名称
    const modelName = model.name || model.model_type || 
      (model.path ? model.path.split('/').pop().replace('.joblib', '') : '');
    
    try {
      // 尝试从公共目录加载模型指标
      const response = await fetch(`/data/${modelName}_metrics.json`);
      
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      } else {
        console.error('无法加载模型指标:', response.statusText);
      }
    } catch (error) {
      console.error('加载模型指标出错:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // 组件加载时获取指标
  useEffect(() => {
    if (showDetails) {
      loadModelMetrics();
    }
  }, [model, showDetails]);
  
  // 处理激活模型的点击事件
  const handleActivate = async () => {
    if (onActivate) {
      onActivate(model);
    }
  };
  
  // 打开详情模态框
  const openDetailsModal = () => {
    if (!metrics) {
      loadModelMetrics();
    }
    setModalVisible(true);
  };
  
  // 准备特征重要性数据
  const getFeatureImportanceData = () => {
    if (!metrics || !metrics.feature_importance) return [];
    
    return metrics.feature_importance.map((item, index) => ({
      key: index,
      feature: item.feature,
      importance: item.importance,
      percentage: (item.importance * 100).toFixed(2)
    }));
  };
  
  // 模型卡片样式
  const cardStyle = {
    marginBottom: 16,
    borderColor: isActive ? '#1890ff' : undefined,
    boxShadow: isActive ? '0 0 8px rgba(24, 144, 255, 0.5)' : undefined
  };
  
  // 获取模型名称显示
  const getModelName = () => {
    const name = model.model_type || model.name || 
      (model.path ? model.path.split('/').pop().replace('.joblib', '') : '未知模型');
    return name;
  };
  
  return (
    <>
      <Card 
        title={
          <Space>
            <ExperimentOutlined />
            <span>{getModelName()}</span>
            {isActive && <Tag color="blue" icon={<CheckCircleOutlined />}>当前活跃</Tag>}
          </Space>
        }
        extra={
          <Space>
            <Button 
              type="primary" 
              size="small"
              onClick={handleActivate}
              disabled={isActive}
            >
              {isActive ? '已激活' : '激活模型'}
            </Button>
            <Button 
              type="link" 
              size="small"
              icon={<InfoCircleOutlined />}
              onClick={openDetailsModal}
            >
              详情
            </Button>
          </Space>
        }
        style={cardStyle}
      >
        <Spin spinning={loading}>
          {metrics ? (
            <>
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="R²得分"
                    value={metrics.r2_score}
                    precision={4}
                    valueStyle={{ color: metrics.r2_score > 0.7 ? '#3f8600' : metrics.r2_score > 0.5 ? '#faad14' : '#cf1322' }}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="RMSE"
                    value={metrics.rmse}
                    precision={2}
                    formatter={value => formatNumber(value)}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="MAE"
                    value={metrics.mae}
                    precision={2}
                    formatter={value => formatNumber(value)}
                  />
                </Col>
              </Row>
              
              <Divider>错误分布</Divider>
              
              {metrics.error_distribution && metrics.error_distribution.error_ranges ? (
                <Row gutter={[8, 16]}>
                  {Object.entries(metrics.error_distribution.error_ranges).map(([range, value]) => (
                    <Col span={8} key={range}>
                      <Tooltip title={`${(value * 100).toFixed(2)}% 的预测在 ${range} 误差范围内`}>
                        <div>
                          <Text>{range} 误差</Text>
                          <Progress 
                            percent={(value * 100).toFixed(2)} 
                            size="small"
                            status={
                              range === "<5%" || range === "5-10%" 
                                ? "success" 
                                : range === "10-15%" 
                                  ? "normal" 
                                  : "exception"
                            }
                          />
                        </div>
                      </Tooltip>
                    </Col>
                  ))}
                </Row>
              ) : (
                <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="无错误分布数据" />
              )}
              
              {metrics.feature_importance && metrics.feature_importance.length > 0 && (
                <>
                  <Divider>主要特征</Divider>
                  {metrics.feature_importance.slice(0, 3).map((feature, index) => (
                    <Row key={index} style={{ marginBottom: 8 }}>
                      <Col span={12}>
                        <Text>{feature.feature.replace(/_/g, ' ')}</Text>
                      </Col>
                      <Col span={12}>
                        <Progress 
                          percent={(feature.importance * 100).toFixed(2)} 
                          size="small"
                        />
                      </Col>
                    </Row>
                  ))}
                </>
              )}
            </>
          ) : (
            <Empty 
              image={Empty.PRESENTED_IMAGE_SIMPLE} 
              description={loading ? "加载中..." : "无评估数据"}
            />
          )}
          
          {model.description && (
            <>
              <Divider />
              <Paragraph ellipsis={{ rows: 2 }}>
                {model.description}
              </Paragraph>
            </>
          )}
        </Spin>
      </Card>
      
      {/* 详细信息模态框 */}
      <Modal
        title={
          <Space>
            <BarChartOutlined />
            <span>{getModelName()} 详细评估指标</span>
          </Space>
        }
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            关闭
          </Button>,
          <Button 
            key="activate" 
            type="primary" 
            onClick={handleActivate}
            disabled={isActive}
          >
            {isActive ? '已激活' : '激活此模型'}
          </Button>
        ]}
        width={800}
      >
        <Spin spinning={loading}>
          {metrics ? (
            <>
              <Title level={4}>性能指标</Title>
              <Row gutter={[16, 16]}>
                <Col span={6}>
                  <Statistic
                    title="R²得分"
                    value={metrics.r2_score}
                    precision={4}
                    valueStyle={{ color: metrics.r2_score > 0.7 ? '#3f8600' : metrics.r2_score > 0.5 ? '#faad14' : '#cf1322' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="RMSE"
                    value={metrics.rmse}
                    precision={2}
                    formatter={value => formatNumber(value)}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="MAE"
                    value={metrics.mae}
                    precision={2}
                    formatter={value => formatNumber(value)}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="MSE"
                    value={metrics.mse}
                    precision={2}
                    formatter={value => formatNumber(value)}
                  />
                </Col>
              </Row>
              
              <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                <Col span={8}>
                  <Statistic
                    title="中位数百分比误差"
                    value={metrics.median_percentage_error}
                    precision={2}
                    suffix="%"
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="平均百分比误差"
                    value={metrics.mean_percentage_error}
                    precision={2}
                    suffix="%"
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="解释方差"
                    value={metrics.explained_variance}
                    precision={4}
                  />
                </Col>
              </Row>
              
              <Divider>错误分布</Divider>
              
              {metrics.error_distribution && metrics.error_distribution.percentiles && (
                <>
                  <Title level={5}>百分位数</Title>
                  <Row gutter={[16, 16]}>
                    <Col span={4}>
                      <Statistic
                        title="P10"
                        value={metrics.error_distribution.percentiles.p10}
                        precision={2}
                        suffix="%"
                      />
                    </Col>
                    <Col span={4}>
                      <Statistic
                        title="P25"
                        value={metrics.error_distribution.percentiles.p25}
                        precision={2}
                        suffix="%"
                      />
                    </Col>
                    <Col span={4}>
                      <Statistic
                        title="P50 (中位数)"
                        value={metrics.error_distribution.percentiles.p50}
                        precision={2}
                        suffix="%"
                      />
                    </Col>
                    <Col span={4}>
                      <Statistic
                        title="P75"
                        value={metrics.error_distribution.percentiles.p75}
                        precision={2}
                        suffix="%"
                      />
                    </Col>
                    <Col span={4}>
                      <Statistic
                        title="P90"
                        value={metrics.error_distribution.percentiles.p90}
                        precision={2}
                        suffix="%"
                      />
                    </Col>
                  </Row>
                </>
              )}
              
              {metrics.error_distribution && metrics.error_distribution.error_ranges && (
                <>
                  <Title level={5} style={{ marginTop: 16 }}>误差范围分布</Title>
                  <Row gutter={[16, 16]}>
                    {Object.entries(metrics.error_distribution.error_ranges).map(([range, value]) => (
                      <Col span={8} key={range}>
                        <Statistic
                          title={`${range} 误差范围内样本比例`}
                          value={value * 100}
                          precision={2}
                          suffix="%"
                          valueStyle={{ 
                            color: range === "<5%" 
                              ? '#3f8600' 
                              : range === "5-10%" 
                                ? '#52c41a' 
                                : range === "10-15%" 
                                  ? '#faad14' 
                                  : range === "15-20%" 
                                    ? '#fa8c16' 
                                    : '#cf1322'
                          }}
                        />
                      </Col>
                    ))}
                  </Row>
                </>
              )}
              
              {metrics.feature_importance && metrics.feature_importance.length > 0 && (
                <>
                  <Divider>特征重要性</Divider>
                  <Table
                    dataSource={getFeatureImportanceData()}
                    pagination={false}
                    size="small"
                    scroll={{ y: 300 }}
                  >
                    <Table.Column 
                      title="特征" 
                      dataIndex="feature" 
                      key="feature"
                      render={text => text.replace(/_/g, ' ')}
                    />
                    <Table.Column 
                      title="重要性" 
                      dataIndex="importance" 
                      key="importance"
                      render={value => value.toFixed(4)}
                    />
                    <Table.Column 
                      title="占比" 
                      dataIndex="percentage" 
                      key="percentage"
                      render={value => (
                        <Progress 
                          percent={value} 
                          size="small"
                          format={percent => `${percent}%`}
                        />
                      )}
                    />
                  </Table>
                </>
              )}
              
              {metrics.parameters && (
                <>
                  <Divider>模型参数</Divider>
                  <Card size="small">
                    <pre style={{ maxHeight: 200, overflow: 'auto' }}>
                      {JSON.stringify(metrics.parameters, null, 2)}
                    </pre>
                  </Card>
                </>
              )}
            </>
          ) : (
            <Empty 
              image={Empty.PRESENTED_IMAGE_SIMPLE} 
              description={loading ? "加载中..." : "无法加载模型评估指标"}
            />
          )}
        </Spin>
      </Modal>
    </>
  );
};

export default ModelCard; 