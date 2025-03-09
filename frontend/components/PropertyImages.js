import React from 'react';
import { Card, Row, Col, Typography } from 'antd';

const { Text } = Typography;

const PropertyImages = () => {
  // 图片类型数组
  const imageTypes = [
    { title: '房产照片示意图' },
    { title: '室内照片示意图' },
    { title: '小区照片示意图' },
    { title: '周边环境示意图' }
  ];
  
  return (
    <Card title="房产照片" style={{ marginTop: '24px' }} bordered={false}>
      <Row gutter={[16, 16]}>
        {imageTypes.map((item, index) => (
          <Col key={index} xs={24} sm={12} md={8} lg={6}>
            <div style={{ 
              background: '#f0f5ff', 
              height: '200px', 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              marginBottom: '16px',
              borderRadius: '4px'
            }}>
              <Text type="secondary">{item.title}</Text>
            </div>
          </Col>
        ))}
      </Row>
    </Card>
  );
};

export default PropertyImages; 