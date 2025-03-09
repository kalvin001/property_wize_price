'use client';

import React from 'react';
import { Card, Typography, Progress, Row, Col, Statistic, Tooltip, Spin } from 'antd';
import { InfoCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import dynamic from 'next/dynamic';

// 使用动态导入并禁用 SSR
const Gauge = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Gauge),
  { ssr: false, loading: () => <div style={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Paragraph, Text } = Typography;

const PropertyPriceRange = ({ priceRange, confidenceInterval, modelExplanation }) => {
  if (!priceRange || !confidenceInterval || !modelExplanation) {
    return null;
  }

  // 配置价格区间仪表盘
  const gaugeConfig = {
    percent: 0.75,
    range: {
      color: 'l(0) 0:#B8E1FF 1:#3D76DD',
    },
    startAngle: Math.PI * 1.2,
    endAngle: Math.PI * -0.2,
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    statistic: {
      content: {
        formatter: () => `A$${(priceRange.most_likely / 10000).toFixed(1)}万`,
        style: {
          fontSize: '24px',
          lineHeight: '36px',
          color: '#1890ff',
        },
      },
      title: {
        content: '最可能价格',
        style: {
          fontSize: '14px',
          lineHeight: '20px',
        },
      },
    },
    axis: {
      label: {
        formatter: (v) => {
          if (v === '0%') return `A$${(priceRange.min / 10000).toFixed(0)}万`;
          if (v === '100%') return `A$${(priceRange.max / 10000).toFixed(0)}万`;
          return '';
        },
      },
      tickLine: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
  };

  return (
    <Card title="价格区间与置信度分析" style={{ marginTop: '24px' }} bordered={false}>
      <Row gutter={[24, 24]}>
        <Col xs={24} md={12}>
          <Card title="价格预测区间" bordered={false}>
            <div style={{ height: '250px' }}>
              <Gauge {...gaugeConfig} />
            </div>
            <Paragraph>
              <InfoCircleOutlined style={{ marginRight: '8px' }} />
              根据模型预测，该房产的价格区间在 
              <Text strong>A${(priceRange.min / 10000).toFixed(0)}万</Text> 至 
              <Text strong>A${(priceRange.max / 10000).toFixed(0)}万</Text> 之间，
              最可能的价格为 <Text strong>A${(priceRange.most_likely / 10000).toFixed(1)}万</Text>。
            </Paragraph>
          </Card>
        </Col>
        
        <Col xs={24} md={12}>
          <Card 
            title={
              <span>
                模型置信度分析 
                <Tooltip title="置信度基于模型精度、数据质量和可比房产相似度综合计算得出">
                  <QuestionCircleOutlined style={{ marginLeft: '8px' }} />
                </Tooltip>
              </span>
            } 
            bordered={false}
          >
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
              <Statistic
                title="模型预测置信度"
                value={modelExplanation.prediction_confidence}
                suffix="%"
                valueStyle={{ color: '#3f8600' }}
              />
              <Progress 
                percent={modelExplanation.prediction_confidence} 
                status="active" 
                strokeColor={{ 
                  from: '#108ee9',
                  to: '#87d068',
                }}
                style={{ marginTop: '16px' }}
              />
            </div>
            
            <Paragraph>
              <Text strong>置信区间（{confidenceInterval.confidence_level * 100}%）：</Text>
              <br />
              <Text>下限：A${(confidenceInterval.lower_bound / 10000).toFixed(1)}万</Text>
              <br />
              <Text>上限：A${(confidenceInterval.upper_bound / 10000).toFixed(1)}万</Text>
            </Paragraph>
            
            <Paragraph>
              <InfoCircleOutlined style={{ marginRight: '8px' }} />
              模型采用了 <Text strong>{modelExplanation.model_type}</Text>，
              模型 R² 分数为 <Text strong>{modelExplanation.r2_score}</Text>，
              平均绝对误差率为 <Text strong>{modelExplanation.mape}%</Text>。
            </Paragraph>
          </Card>
        </Col>
      </Row>
    </Card>
  );
};

export default PropertyPriceRange; 