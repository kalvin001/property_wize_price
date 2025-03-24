import { useState, useEffect } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { 
  Spin, 
  Typography, 
  Card, 
  Row, 
  Col, 
  Divider, 
  Statistic, 
  Table, 
  Button,
  Tag,
  Descriptions,
  Progress,
  Alert,
  List,
  Select,
  Space,
  message
} from 'antd';
import {
  DownloadOutlined,
  BarChartOutlined,
  FileTextOutlined,
  HomeOutlined
} from '@ant-design/icons';
import './PropertyDetail.css';

const { Title, Paragraph, Text } = Typography;
const { Option } = Select;

// 解析URL参数的辅助函数
function useQuery() {
  return new URLSearchParams(useLocation().search);
}

const PropertyDetail = () => {
  const { propId } = useParams();
  const query = useQuery();
  const modelFromUrl = query.get('model');
  
  const [loading, setLoading] = useState(true);
  const [property, setProperty] = useState(null);
  const [modelOptions, setModelOptions] = useState([]);
  const [selectedModel, setSelectedModel] = useState(modelFromUrl);
  
  // 获取所有可用模型
  const fetchModelOptions = async () => {
    try {
      const response = await fetch('/api/models');
      if (response.ok) {
        const data = await response.json();
        const options = data.models.map(model => ({
          label: model.model_type || model.name || model.path.split('/').pop().replace('.joblib', ''),
          value: model.path.split('/').pop()
        }));
        setModelOptions(options);
      }
    } catch (error) {
      console.error('获取模型列表失败:', error);
    }
  };
  
  // 加载房产详情
  const fetchPropertyDetail = async () => {
    setLoading(true);
    
    try {
      let url = `/api/properties/${propId}`;
      
      // 如果有选择模型，添加到请求URL
      if (selectedModel) {
        url += `?model=${encodeURIComponent(selectedModel)}`;
      }
      
      const response = await fetch(url);
      
      if (response.ok) {
        const data = await response.json();
        setProperty(data);
      } else {
        message.error('获取房产详情失败');
      }
    } catch (error) {
      console.error('获取房产详情出错:', error);
      message.error('获取房产详情出错');
    } finally {
      setLoading(false);
    }
  };
  
  // 初始化加载
  useEffect(() => {
    fetchModelOptions();
  }, []);
  
  // 当propId或selectedModel变化时重新加载数据
  useEffect(() => {
    if (propId) {
      fetchPropertyDetail();
    }
  }, [propId, selectedModel]);
  
  // 下载PDF报告
  const downloadReport = async () => {
    try {
      let url = `/api/properties/${propId}/pdf`;
      
      // 如果有选择模型，添加到请求URL
      if (selectedModel) {
        url += `?model=${encodeURIComponent(selectedModel)}`;
      }
      
      window.open(url, '_blank');
    } catch (error) {
      message.error('下载报告失败');
    }
  };
  
  // 模型变更处理
  const handleModelChange = (value) => {
    setSelectedModel(value);
    
    // 更新URL但不重新加载页面
    const url = new URL(window.location.href);
    if (value) {
      url.searchParams.set('model', value);
    } else {
      url.searchParams.delete('model');
    }
    window.history.pushState({}, '', url);
  };
  
  if (loading) {
    return (
      <div className="loading-container">
        <Spin size="large" />
        <Typography.Text>正在加载房产详情...</Typography.Text>
      </div>
    );
  }
  
  if (!property) {
    return (
      <div className="error-container">
        <Alert
          message="未找到房产"
          description="无法找到指定ID的房产信息"
          type="error"
          showIcon
        />
        <Button type="primary" href="/valuation">返回列表</Button>
      </div>
    );
  }
  
  // 准备特征重要性数据
  const featureImportanceData = property.feature_importance.map((item, index) => ({
    key: index,
    name: item.feature,
    importance: item.importance,
    value: item.value || 0,
    percentage: (item.importance * 100).toFixed(2)
  }));
  
  return (
    <div className="property-detail-container">
      <div className="page-header">
        <div>
          <Title level={2}>
            <HomeOutlined /> {property.address}
          </Title>
          <Text type="secondary">ID: {property.prop_id}</Text>
        </div>
        <div className="header-actions">
          <Space>
            <Select
              placeholder="选择模型"
              allowClear
              style={{ width: 180 }}
              value={selectedModel}
              onChange={handleModelChange}
            >
              {modelOptions.map(option => (
                <Option key={option.value} value={option.value}>
                  {option.label}
                </Option>
              ))}
            </Select>
            <Button type="primary" onClick={downloadReport} icon={<DownloadOutlined />}>
              下载估价报告
            </Button>
            <Button href="/valuation">返回列表</Button>
          </Space>
        </div>
      </div>
      
      {selectedModel && (
        <Alert 
          message={
            <Space>
              <span>当前使用模型: </span>
              <Tag color="blue">
                {modelOptions.find(m => m.value === selectedModel)?.label || selectedModel}
              </Tag>
            </Space>
          }
          type="info" 
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}
      
      <Row gutter={[16, 16]}>
        <Col xs={24} md={8}>
          <Card title="估价结果" bordered={false}>
            <Statistic
              title="预测价格"
              value={property.predicted_price}
              precision={2}
              prefix="¥"
              formatter={value => `${value.toLocaleString()}`}
              style={{ marginBottom: 16 }}
            />
            
            <Divider orientation="left">价格区间</Divider>
            <Row gutter={16}>
              <Col span={12}>
                <Statistic 
                  title="最低价格" 
                  value={property.price_range?.min} 
                  precision={2} 
                  prefix="¥"
                  formatter={value => `${value.toLocaleString()}`}
                  valueStyle={{ fontSize: '14px' }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="最高价格" 
                  value={property.price_range?.max} 
                  precision={2} 
                  prefix="¥"
                  formatter={value => `${value.toLocaleString()}`}
                  valueStyle={{ fontSize: '14px' }}
                />
              </Col>
            </Row>
            
            <Divider orientation="left">置信区间</Divider>
            <Statistic 
              title="90%置信区间" 
              value={`¥${property.confidence_interval?.lower.toLocaleString()} - ¥${property.confidence_interval?.upper.toLocaleString()}`}
              valueStyle={{ fontSize: '14px' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} md={16}>
          <Card title="房产特征" bordered={false}>
            <Descriptions bordered column={{ xxl: 4, xl: 3, lg: 3, md: 2, sm: 1, xs: 1 }}>
              {property.features && Object.entries(property.features)
                .filter(([key]) => !['prop_id', 'model_name'].includes(key))
                .map(([key, value]) => (
                  <Descriptions.Item 
                    key={key} 
                    label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  >
                    {typeof value === 'number' ? value.toLocaleString() : value}
                  </Descriptions.Item>
                ))
              }
            </Descriptions>
          </Card>
        </Col>
      </Row>
      
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <BarChartOutlined />
                <span>价格影响因素</span>
              </Space>
            } 
            bordered={false}
          >
            <Table 
              dataSource={featureImportanceData.slice(0, 10)} 
              pagination={false}
              rowKey="key"
            >
              <Table.Column title="特征" dataIndex="name" key="name" 
                render={text => text.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              />
              <Table.Column title="数值" dataIndex="value" key="value" 
                render={value => typeof value === 'number' ? value.toLocaleString() : value}
              />
              <Table.Column title="重要性" dataIndex="percentage" key="percentage" 
                render={percentage => (
                  <Progress 
                    percent={percentage} 
                    size="small" 
                    format={percent => `${percent}%`}
                  />
                )}
              />
            </Table>
          </Card>
        </Col>
        
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <FileTextOutlined />
                <span>AI解释</span>
              </Space>
            } 
            bordered={false}
          >
            {property.ai_explanation && property.ai_explanation.factors && (
              <>
                <Paragraph>{property.ai_explanation.summary}</Paragraph>
                <Divider />
                <List
                  size="small"
                  header={<div><strong>主要影响因素</strong></div>}
                  bordered
                  dataSource={property.ai_explanation.factors}
                  renderItem={item => (
                    <List.Item>
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <div>{item.description}</div>
                        <div>
                          <Tag color={item.impact > 0 ? 'green' : 'red'}>
                            {item.impact > 0 ? '+' : ''}{item.impact}%
                          </Tag>
                        </div>
                      </div>
                    </List.Item>
                  )}
                />
              </>
            )}
          </Card>
        </Col>
      </Row>
      
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="可比较房产" bordered={false}>
            <Table 
              dataSource={property.comparable_properties} 
              rowKey={(record) => record.prop_id || record.id || Math.random().toString()}
              pagination={false}
            >
              <Table.Column title="地址" dataIndex="address" key="address" />
              <Table.Column title="价格" dataIndex="price" key="price" 
                render={price => `¥${price.toLocaleString()}`}
              />
              <Table.Column title="与目标差异" dataIndex="price_diff_percent" key="price_diff_percent" 
                render={diff => (
                  <Tag color={diff >= 0 ? 'green' : 'red'}>
                    {diff >= 0 ? '+' : ''}{diff}%
                  </Tag>
                )}
              />
              <Table.Column title="相似度" dataIndex="similarity" key="similarity" 
                render={similarity => (
                  <Progress 
                    percent={similarity * 100} 
                    size="small"
                    format={percent => `${percent.toFixed(0)}%`}
                  />
                )}
              />
            </Table>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default PropertyDetail; 