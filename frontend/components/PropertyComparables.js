import React from 'react';
import { Card, Table, Empty, Typography, Tag, Progress, Tooltip, Divider, Row, Col, Statistic } from 'antd';
import { InfoCircleOutlined, EnvironmentOutlined, HomeOutlined, DollarCircleOutlined } from '@ant-design/icons';

const { Paragraph, Text } = Typography;

const PropertyComparables = ({ comparableProperties }) => {
  // 找出当前房产和可比房产
  const currentProperty = comparableProperties?.find(p => p.is_current) || null;
  const otherProperties = comparableProperties?.filter(p => !p.is_current) || [];
  
  // 计算价格区间和平均价格
  const calculatePriceStats = () => {
    if (!otherProperties || otherProperties.length === 0) return null;
    
    const prices = otherProperties.map(p => p.price).filter(p => p > 0);
    if (prices.length === 0) return null;
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
    
    // 计算当前房产价格与平均价格的差异
    const priceDiff = currentProperty?.price ? (currentProperty.price - avgPrice) / avgPrice * 100 : 0;
    
    return {
      minPrice,
      maxPrice,
      avgPrice,
      priceDiff
    };
  };
  
  const priceStats = calculatePriceStats();
  
  const comparableColumns = [
    {
      title: '可比房产',
      dataIndex: 'address',
      key: 'address',
      ellipsis: true,
      render: (address, record) => (
        <span>
          {record.is_current && <HomeOutlined style={{ color: '#1890ff', marginRight: '8px' }} />}
          <Tooltip title={address}>
            {address}
          </Tooltip>
        </span>
      ),
      responsive: ['md'],
    },
    {
      title: '价格(万元)',
      dataIndex: 'price',
      key: 'price',
      render: (price, record) => (
        <span style={{ color: record.is_current ? '#1890ff' : 'inherit', fontWeight: record.is_current ? 'bold' : 'normal' }}>
          {price > 0 ? `A$${(price / 10000).toFixed(1)}` : '暂无价格'}
        </span>
      ),
    },
    {
      title: '面积',
      dataIndex: 'area',
      key: 'area',
      render: (area, record) => (
        <span style={{ color: record.is_current ? '#1890ff' : 'inherit' }}>
          {area > 0 ? `${area.toFixed(1)}m²` : '0m²'}
        </span>
      ),
      responsive: ['md'],
    },
    {
      title: '单价',
      dataIndex: 'unit_price',
      key: 'unit_price',
      render: (unit_price, record) => {
        // 使用后端提供的单价，或前端计算
        const price = record.price || 0;
        const area = record.area || 1; // 防止除以0
        const finalUnitPrice = unit_price || (price / area);
        return (
          <span style={{ color: record.is_current ? '#1890ff' : 'inherit' }}>
            {finalUnitPrice > 0 ? `A$${finalUnitPrice.toFixed(0)}/m²` : '暂无数据'}
          </span>
        );
      },
      responsive: ['md'],
    },
    {
      title: '距离',
      dataIndex: 'distance_km',
      key: 'distance_km',
      render: (distance_km, record) => {
        if (record.is_current) {
          return <Tag color="blue">当前房产</Tag>;
        }
        
        // 处理距离展示，确保不会出现0公里
        let distanceDisplay = '未知距离';
        if (distance_km !== undefined && distance_km !== null) {
          if (distance_km < 0.01) {
            distanceDisplay = '<0.01公里';
          } else if (distance_km < 1) {
            // 小于1公里用米展示
            distanceDisplay = `${Math.round(distance_km * 1000)}米`;
          } else {
            // 大于等于1公里用公里展示，保留1位小数
            distanceDisplay = `${distance_km.toFixed(1)}公里`;
          }
        }
        
        return (
          <span>
            <EnvironmentOutlined style={{ marginRight: '4px' }} />
            {distanceDisplay}
          </span>
        );
      },
      responsive: ['md'],
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => {
        if (record.is_current) {
          return <Tag color="blue">当前房产</Tag>;
        }
        return (
          <Tag color={
            status === '已成交' ? 'green' : 
            status === '在售' ? 'orange' : 'default'
          }>
            {status}
          </Tag>
        );
      },
    },
    {
      title: '相似度',
      dataIndex: 'similarity',
      key: 'similarity',
      render: (value, record) => {
        if (record.is_current) {
          return <Progress percent={100} size="small" status="success" />;
        }
        return (
          <Progress 
            percent={value} 
            size="small"
            format={(percent) => `${percent}%`}
            status="active"
          />
        );
      },
    },
  ];

  // 价格分析区域组件
  const PriceAnalysis = () => {
    if (!priceStats || !currentProperty) return null;
    
    const { minPrice, maxPrice, avgPrice, priceDiff } = priceStats;
    const currentPrice = currentProperty.price;
    
    return (
      <Card 
        title={<><DollarCircleOutlined /> 价格分析</>} 
        style={{ marginBottom: '24px' }}
        bordered={false}
      >
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={8}>
            <Statistic 
              title="当前房产"
              value={`A$${(currentPrice / 10000).toFixed(1)}万`}
              valueStyle={{ color: '#1890ff', fontWeight: 'bold' }}
            />
          </Col>
          <Col xs={24} sm={8}>
            <Statistic 
              title="可比房产价格区间"
              value={`A$${(minPrice / 10000).toFixed(1)} - ${(maxPrice / 10000).toFixed(1)}万`}
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col xs={24} sm={8}>
            <Statistic 
              title="与平均价格对比"
              value={`${priceDiff > 0 ? '+' : ''}${priceDiff.toFixed(1)}%`}
              valueStyle={{ color: priceDiff > 0 ? '#f5222d' : '#52c41a' }}
              suffix={
                <Tooltip title={`可比房产平均价格: A$${(avgPrice / 10000).toFixed(1)}万`}>
                  <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                </Tooltip>
              }
            />
            <Text type="secondary">
              {priceDiff > 0 
                ? '高于平均价格' 
                : priceDiff < 0 
                  ? '低于平均价格'
                  : '与平均价格相当'}
            </Text>
          </Col>
        </Row>
      </Card>
    );
  };

  return (
    <>
      {(currentProperty && otherProperties.length > 0) && <PriceAnalysis />}
      
      <Card title="可比房产" style={{ marginTop: '24px' }} bordered={false}>
        <Paragraph style={{ marginBottom: '16px' }}>
          <InfoCircleOutlined style={{ marginRight: '8px' }} />
          以下房产是基于地理位置、建筑特征和价格趋势筛选出的最相似房产，作为参考依据。
          {currentProperty && (
            <Tag color="blue" style={{ marginLeft: '8px' }}>
              <HomeOutlined /> 蓝色标记为当前房产
            </Tag>
          )}
        </Paragraph>
        
        {comparableProperties && comparableProperties.length > 0 ? (
          <Table 
            columns={comparableColumns} 
            dataSource={comparableProperties}
            pagination={false}
            rowKey="id"
            scroll={{ x: 'max-content' }}
            rowClassName={(record) => record.is_current ? 'current-property-row' : ''}
          />
        ) : (
          <Empty description="暂无可比房产数据" />
        )}
      </Card>
      
      <style jsx global>{`
        .current-property-row {
          background-color: #e6f7ff;
        }
        .current-property-row:hover td {
          background-color: #bae7ff !important;
        }
      `}</style>
    </>
  );
};

export default PropertyComparables; 