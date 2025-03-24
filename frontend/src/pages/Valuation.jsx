import { useState, useEffect } from 'react';
import {
  Typography, 
  Table, 
  Card, 
  Row, 
  Col, 
  Divider, 
  Statistic, 
  Spin, 
  message, 
  Button, 
  Pagination,
  Input,
  Select,
  Tag,
  Tooltip,
  Space
} from 'antd';

const { Search } = Input;
const { Option } = Select;

const Valuation = () => {
  const [loading, setLoading] = useState(false);
  const [properties, setProperties] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(3);
  const [searchQuery, setSearchQuery] = useState('');
  const [modelOptions, setModelOptions] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  
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

  useEffect(() => {
    fetchModelOptions();
  }, []);

  // 加载房产数据
  const fetchProperties = async () => {
    setLoading(true);
    
    try {
      let url = `/api/properties?page=${page}&page_size=${pageSize}`;
      
      // 添加搜索查询参数
      if (searchQuery) {
        url += `&query=${encodeURIComponent(searchQuery)}`;
      }
      
      // 添加模型选择参数
      if (selectedModel) {
        url += `&model=${encodeURIComponent(selectedModel)}`;
      }
      
      const response = await fetch(url);
      
      if (response.ok) {
        const data = await response.json();
        setProperties(data.properties);
        setTotal(data.total);
      } else {
        message.error('获取房产数据失败');
      }
    } catch (error) {
      console.error('获取房产数据出错:', error);
      message.error('获取房产数据出错');
    } finally {
      setLoading(false);
    }
  };

  // 当页码、每页数量或搜索条件变化时重新加载数据
  useEffect(() => {
    fetchProperties();
  }, [page, pageSize, searchQuery, selectedModel]);
  
  return (
    <div className="valuation-container">
      <Typography.Title level={2}>房产估值</Typography.Title>
      
      <Card className="filter-card">
        <Row gutter={16} align="middle">
          <Col xs={24} sm={12} md={8} lg={6}>
            <Search
              placeholder="输入地址或ID搜索"
              allowClear
              onSearch={(value) => {
                setSearchQuery(value);
                setPage(1); // 重置页码
              }}
              style={{ width: '100%' }}
            />
          </Col>
          <Col xs={24} sm={12} md={8} lg={6}>
            <Select
              placeholder="选择模型"
              allowClear
              style={{ width: '100%' }}
              onChange={(value) => {
                setSelectedModel(value);
                setPage(1); // 重置页码
              }}
            >
              {modelOptions.map(option => (
                <Option key={option.value} value={option.value}>
                  {option.label}
                </Option>
              ))}
            </Select>
          </Col>
          <Col xs={24} sm={24} md={8} lg={6}>
            <Space>
              {selectedModel && (
                <Tag color="blue">
                  使用模型: {modelOptions.find(m => m.value === selectedModel)?.label || selectedModel}
                </Tag>
              )}
              <Button 
                type="primary" 
                onClick={fetchProperties}
              >
                刷新数据
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>
      
      {/* 房产卡片列表 */}
      <div className="properties-container">
        <Spin spinning={loading}>
          {properties.length > 0 ? (
            <Row gutter={[16, 16]}>
              {properties.map(property => (
                <Col xs={24} sm={24} md={12} lg={8} key={property.prop_id}>
                  <Card
                    hoverable
                    className="property-card"
                    onClick={() => window.location.href = `/property/${property.prop_id}${selectedModel ? `?model=${selectedModel}` : ''}`}
                  >
                    <div className="property-address">
                      <Typography.Title level={4}>{property.address}</Typography.Title>
                      <Typography.Text type="secondary">ID: {property.prop_id}</Typography.Text>
                    </div>
                    <Divider />
                    <Row gutter={16}>
                      <Col span={24}>
                        <Statistic 
                          title="预测价格" 
                          value={property.predicted_price} 
                          precision={2} 
                          prefix="¥"
                          formatter={value => `${value.toLocaleString()}`}
                        />
                      </Col>
                    </Row>
                    <Divider />
                    <Row gutter={8}>
                      {property.features && (
                        <>
                          {property.features.bedrooms && (
                            <Col span={8}>
                              <Statistic title="卧室" value={property.features.bedrooms} />
                            </Col>
                          )}
                          {property.features.bathrooms && (
                            <Col span={8}>
                              <Statistic title="浴室" value={property.features.bathrooms} />
                            </Col>
                          )}
                          {property.features.prop_area && (
                            <Col span={8}>
                              <Statistic title="面积(m²)" value={property.features.prop_area} />
                            </Col>
                          )}
                        </>
                      )}
                    </Row>
                    <div className="property-footer">
                      <Button type="primary" size="small">
                        查看详情
                      </Button>
                      {selectedModel && (
                        <Tooltip title={`使用模型: ${modelOptions.find(m => m.value === selectedModel)?.label || selectedModel}`}>
                          <Tag color="blue">模型预测</Tag>
                        </Tooltip>
                      )}
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          ) : (
            <div className="empty-state">
              <Typography.Text>没有找到房产数据</Typography.Text>
            </div>
          )}
          
          {/* 分页控件 */}
          {total > 0 && (
            <div className="pagination-container">
              <Pagination
                current={page}
                pageSize={pageSize}
                total={total}
                onChange={(newPage, newPageSize) => {
                  setPage(newPage);
                  if (newPageSize !== pageSize) {
                    setPageSize(newPageSize);
                  }
                }}
                showSizeChanger
                showQuickJumper
                showTotal={total => `共 ${total} 条记录`}
              />
            </div>
          )}
        </Spin>
      </div>
    </div>
  );
};

export default Valuation; 