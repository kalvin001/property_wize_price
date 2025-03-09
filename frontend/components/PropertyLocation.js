import React from 'react';
import { Card, Row, Col, Statistic, Typography } from 'antd';

const { Text } = Typography;

const PropertyLocation = ({ latitude, longitude }) => {
  // 检查坐标是否有效
  const hasValidCoordinates = latitude > 0 || longitude > 0;
  
  if (!hasValidCoordinates) return null;
  
  return (
    <Card title="地理位置" style={{ marginTop: '24px' }} bordered={false}>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12}>
          <Statistic title="经度" value={longitude.toFixed(6)} />
        </Col>
        <Col xs={24} sm={12}>
          <Statistic title="纬度" value={latitude.toFixed(6)} />
        </Col>
      </Row>
      <div style={{ 
        marginTop: '16px', 
        background: '#f0f5ff', 
        height: '200px', 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        borderRadius: '4px'
      }}>
        <Text type="secondary">地图加载中...</Text>
      </div>
    </Card>
  );
};

export default PropertyLocation; 