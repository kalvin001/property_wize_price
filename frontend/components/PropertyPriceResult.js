import React, { useEffect, useState } from 'react';
import { Card, Statistic, Divider, Row, Col } from 'antd';
import { 
  DollarOutlined, 
  ArrowUpOutlined, 
  ArrowDownOutlined 
} from '@ant-design/icons';

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
      
      {property.area > 0 && (
        <div style={{ marginTop: '16px' }}>
          <Statistic
            title="单价"
            value={(property.predicted_price / property.area).toFixed(0)}
            suffix="元/m²"
            valueStyle={{ 
              fontSize: isMobile ? '14px' : '16px',
              lineHeight: 1.4
            }}
          />
        </div>
      )}
    </Card>
  );
};

export default PropertyPriceResult; 