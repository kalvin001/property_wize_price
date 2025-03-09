'use client';

import React from 'react';
import { Card, Typography, Empty, Statistic, Row, Col, Divider, Spin, Space, Tag } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, EnvironmentOutlined, HomeOutlined, BankOutlined } from '@ant-design/icons';
import dynamic from 'next/dynamic';

// 使用动态导入并禁用 SSR
const Column = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Column),
  { ssr: false, loading: () => <div style={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Text, Title, Paragraph } = Typography;

const PropertyNeighborhoodStats = ({ neighborhoodStats }) => {
  if (!neighborhoodStats) {
    return null;
  }

  // 生成周边房价柱状图数据
  const generateNeighborhoodData = () => {
    // 使用后端提供的radius_stats数据
    if (neighborhoodStats.radius_stats && neighborhoodStats.radius_stats.length > 0) {
      return neighborhoodStats.radius_stats.map(stat => ({
        type: `${stat.radius}km内`,
        price: stat.avg_price,
        count: stat.count
      })).concat([
        {
          type: '区域均价',
          price: neighborhoodStats.avg_price,
          count: neighborhoodStats.num_properties
        }
      ]);
    }
    
    // 如果没有radius_stats，使用模拟数据
    return [
      {
        type: '1km内',
        price: neighborhoodStats.avg_price * 1.05,
      },
      {
        type: '2km内',
        price: neighborhoodStats.avg_price,
      },
      {
        type: '3km内',
        price: neighborhoodStats.avg_price * 0.95,
      },
      {
        type: '5km内',
        price: neighborhoodStats.avg_price * 0.9,
      },
      {
        type: '区域均价',
        price: neighborhoodStats.avg_price * 0.92,
      },
    ];
  };

  const neighborhoodData = generateNeighborhoodData();
  
  const columnConfig = {
    data: neighborhoodData,
    xField: 'type',
    yField: 'price',
    columnWidthRatio: 0.6,
    label: {
      position: 'top',
      formatter: (data) => `A$${(data.price / 10000).toFixed(0)}万`,
    },
    yAxis: {
      label: {
        formatter: (v) => `A$${(v / 10000).toFixed(0)}万`,
      },
    },
    color: (data) => {
      if (data.type === '区域均价') return '#faad14';
      return '#1890ff';
    },
    tooltip: {
      formatter: (data) => {
        const tooltipInfo = { name: '均价', value: `A$${(data.price / 10000).toFixed(2)}万` };
        if (data.count) {
          tooltipInfo.count = `${data.count}套房产`;
        }
        return tooltipInfo;
      },
    },
  };

  return (
    <Card title="周边区域房产分析" style={{ marginTop: '24px' }} bordered={false}>
      <Row gutter={[24, 24]}>
        <Col xs={24} md={12}>
          <Title level={5}>区域房价统计</Title>
          <Paragraph>
            该地区共有<Text strong>{neighborhoodStats.num_properties}</Text>套在售房产，
            区域价格呈<Text strong style={{ color: neighborhoodStats.price_trend === '上升' ? '#3f8600' : (neighborhoodStats.price_trend === '下降' ? '#cf1322' : '#1890ff') }}>
              {neighborhoodStats.price_trend}
            </Text>趋势。
          </Paragraph>
          
          <Space direction="vertical" style={{ width: '100%' }}>
            <Statistic
              title="区域平均价格"
              value={neighborhoodStats.avg_price}
              formatter={value => `A$${(value / 10000).toFixed(0)}万`}
              valueStyle={{ color: '#1890ff' }}
              prefix={<HomeOutlined />}
            />
            
            <Statistic
              title="区域平均单价"
              value={neighborhoodStats.avg_price_per_sqm}
              formatter={value => `A$${value.toFixed(0)}/㎡`}
              valueStyle={{ color: '#1890ff' }}
              prefix={<BankOutlined />}
            />
            
            <Divider style={{ margin: '12px 0' }} />
            
            <Space>
              <Tag color="blue" icon={<EnvironmentOutlined />}>最高: A${(neighborhoodStats.max_price / 10000).toFixed(0)}万</Tag>
              <Tag color="orange" icon={<EnvironmentOutlined />}>最低: A${(neighborhoodStats.min_price / 10000).toFixed(0)}万</Tag>
            </Space>
            
            <Paragraph style={{ marginTop: '12px' }}>
              相比区域平均，此房产价格
              {neighborhoodStats.avg_price < neighborhoodStats.avg_price ? (
                <Text type="success"> 低于平均 <ArrowDownOutlined /></Text>
              ) : (
                <Text type="warning"> 高于平均 <ArrowUpOutlined /></Text>
              )}
            </Paragraph>
          </Space>
        </Col>
        
        <Col xs={24} md={12}>
          <Title level={5}>周边不同范围房价对比</Title>
          <div style={{ height: '260px' }}>
            <Column {...columnConfig} />
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default PropertyNeighborhoodStats; 