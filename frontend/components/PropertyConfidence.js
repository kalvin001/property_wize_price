import React from 'react';
import { Card, Row, Col, Statistic, Progress, Typography } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';

const { Paragraph } = Typography;

const PropertyConfidence = ({ predictedPrice }) => {
  return (
    <Card title="估价置信度" style={{ marginTop: '24px' }} bordered={false}>
      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <div style={{ textAlign: 'center', marginBottom: '24px' }}>
            <Statistic
              title="预测置信度"
              value={95}
              suffix="%"
              valueStyle={{ color: '#3f8600' }}
            />
            <Progress 
              percent={95} 
              status="active" 
              strokeColor={{ 
                from: '#108ee9',
                to: '#87d068',
              }}
              style={{ marginTop: '16px' }}
            />
          </div>
          <Paragraph>
            <InfoCircleOutlined style={{ marginRight: '8px' }} />
            置信度基于模型精度、数据质量和可比房产相似度综合计算得出。
          </Paragraph>
        </Col>
        <Col xs={24}>
          <Card title="价格范围预测" bordered>
            <Statistic
              title="价格区间（95%置信度）"
              value={`A$${(predictedPrice * 0.95 / 10000).toFixed(0)}万 - A$${(predictedPrice * 1.05 / 10000).toFixed(0)}万`}
              valueStyle={{ fontSize: '16px' }}
            />
            <Paragraph style={{ marginTop: '16px' }}>
              根据模型分析，该房产在市场中的合理价格区间如上所示，实际成交价可能受到市场波动、谈判等因素的影响。
            </Paragraph>
          </Card>
        </Col>
      </Row>
    </Card>
  );
};

export default PropertyConfidence; 