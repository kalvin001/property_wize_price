import React, { useEffect, useState } from 'react';
import { Card, Statistic, Divider, Row, Col, Typography, Tooltip, Tag } from 'antd';
import { 
  DollarOutlined, 
  ArrowUpOutlined, 
  ArrowDownOutlined,
  HomeOutlined,
  EnvironmentOutlined,
  BankOutlined,
  AreaChartOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Text } = Typography;

const PropertyPriceResult = ({ property }) => {
  const [isMobile, setIsMobile] = useState(false);

  // 检测移动设备
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 576);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => {
      window.removeEventListener('resize', checkMobile);
    };
  }, []);
  
  if (!property) return null;

  // 提取关键特征进行展示
  const renderKeyFeatures = () => {
    if (!property.features) return null;
    
    // 关键特征列表
    const keyFeatures = [
      { key: 'bedrooms', label: '卧室', icon: <HomeOutlined /> },
      { key: 'bathrooms', label: '浴室', icon: <HomeOutlined /> },
      { key: 'car_spaces', label: '车位', icon: <HomeOutlined /> },
      { key: 'land_area', label: '土地面积(m²)', icon: <AreaChartOutlined /> },
      { key: 'building_area', label: '建筑面积(m²)', icon: <AreaChartOutlined /> },
      { key: 'year_built', label: '建造年份', icon: <HomeOutlined /> },
      { key: 'property_type', label: '房产类型', icon: <HomeOutlined /> }
    ];
    
    return (
      <>
        <Divider style={{ margin: '16px 0' }}>房产特征</Divider>
        <Row gutter={[16, 16]}>
          {keyFeatures.map(feature => {
            // 查找特征值(支持多种可能的键名)
            const possibleKeys = [
              feature.key, 
              `property_${feature.key}`, 
              `prop_${feature.key}`
            ];
            
            let value = null;
            for (const key of possibleKeys) {
              if (property.features[key] !== undefined && property.features[key] !== null) {
                value = property.features[key];
                break;
              }
            }
            
            // 如果没有值，跳过这个特征
            if (value === null || value === undefined) return null;
            
            return (
              <Col key={feature.key} xs={12} sm={8} md={8} lg={8}>
                <Tooltip title={feature.label}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ marginRight: '8px' }}>{feature.icon}</span>
                    <div>
                      <Text type="secondary" style={{ fontSize: '12px' }}>{feature.label}</Text>
                      <div>{typeof value === 'number' ? value.toFixed(0) : value}</div>
                    </div>
                  </div>
                </Tooltip>
              </Col>
            );
          })}
        </Row>
        
        {property.features.suburb && (
          <div style={{ marginTop: '16px' }}>
            <Tag color="blue" icon={<EnvironmentOutlined />}>{property.features.suburb || '未知区域'}</Tag>
            {property.features.state && (
              <Tag color="geekblue">{property.features.state}</Tag>
            )}
          </div>
        )}
      </>
    );
  };
  
  return (
    <Card 
      title="估价结果" 
      bordered={false} 
      style={{ marginBottom: '24px' }}
      bodyStyle={{ padding: isMobile ? '16px' : '24px' }}
    >
      <Statistic
        title="AI估价"
        value={property.predicted_price}
        precision={0}
        formatter={value => `A$${(value / 10000).toFixed(0)}万`}
        valueStyle={{ 
          color: '#1890ff', 
          fontSize: isMobile ? '18px' : '24px',
          lineHeight: 1.4
        }}
        prefix={<DollarOutlined />}
      />
      
      {property.actual_price > 0 && (
        <>
          <Divider style={{ margin: '16px 0' }} />
          
          <Row gutter={isMobile ? [0, 16] : 16}>
            <Col xs={24} sm={12}>
              <Statistic
                title="实际价格"
                value={property.actual_price}
                precision={0}
                formatter={value => `A$${(value / 10000).toFixed(0)}万`}
                valueStyle={{ 
                  fontSize: isMobile ? '16px' : '18px',
                  lineHeight: 1.4
                }}
              />
            </Col>
            
            <Col xs={24} sm={12}>
              <Statistic
                title="误差百分比"
                value={property.error_percent}
                precision={1}
                suffix="%"
                valueStyle={{ 
                  color: Math.abs(property.error_percent) < 3 ? '#3f8600' : 
                        Math.abs(property.error_percent) < 5 ? '#faad14' : '#cf1322',
                  fontSize: isMobile ? '16px' : '18px',
                  lineHeight: 1.4
                }}
                prefix={property.error_percent > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
              />
            </Col>
          </Row>
        </>
      )}
      
      <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
        {property.features && property.features.prop_area && (
          <Col xs={24} sm={12}>
            <Statistic
              title="单价"
              value={(property.predicted_price / property.features.prop_area).toFixed(0)}
              suffix="澳元/m²"
              valueStyle={{ 
                fontSize: isMobile ? '14px' : '16px',
                lineHeight: 1.4
              }}
              prefix={<BankOutlined />}
            />
          </Col>
        )}
        
        {property.features && property.features.model_name && (
          <Col xs={24} sm={12}>
            <div>
              <Text type="secondary">使用模型</Text>
              <div style={{ marginTop: '4px' }}>
                <Tag color="purple" icon={<InfoCircleOutlined />}>
                  {property.features.model_name || '默认模型'}
                </Tag>
              </div>
            </div>
          </Col>
        )}
      </Row>
      
      {/* 显示关键房产特征 */}
      {renderKeyFeatures()}
    </Card>
  );
};

export default PropertyPriceResult; 