import React from 'react';
import { Card, Table, Empty, Typography, Tag, Progress } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';

const { Paragraph } = Typography;

const PropertyComparables = ({ comparableProperties }) => {
  const comparableColumns = [
    {
      title: '可比房产',
      dataIndex: 'address',
      key: 'address',
      ellipsis: true,
      responsive: ['md'],
    },
    {
      title: '价格(万元)',
      dataIndex: 'price',
      key: 'price',
      render: (price) => `A$${(price / 10000).toFixed(1)}`,
    },
    {
      title: '面积',
      dataIndex: 'area',
      key: 'area',
      render: (area) => `${area}m²`,
      responsive: ['md'],
    },
    {
      title: '单价',
      dataIndex: 'unit_price',
      key: 'unit_price',
      render: (_, record) => `A$${(record.price / record.area).toFixed(0)}/m²`,
      responsive: ['md'],
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={
          status === '已成交' ? 'green' : 
          status === '在售' ? 'blue' : 'orange'
        }>
          {status}
        </Tag>
      ),
    },
    {
      title: '相似度',
      dataIndex: 'similarity',
      key: 'similarity',
      render: (value) => (
        <Progress 
          percent={value} 
          size="small"
          format={(percent) => `${percent}%`}
          status="active"
        />
      ),
    },
  ];

  return (
    <Card title="可比房产" style={{ marginTop: '24px' }} bordered={false}>
      <Paragraph style={{ marginBottom: '16px' }}>
        <InfoCircleOutlined style={{ marginRight: '8px' }} />
        以下房产是基于地理位置、建筑特征和价格趋势筛选出的最相似房产，作为参考依据。
      </Paragraph>
      {comparableProperties && comparableProperties.length > 0 ? (
        <Table 
          columns={comparableColumns} 
          dataSource={comparableProperties}
          pagination={false}
          rowKey="id"
          scroll={{ x: 'max-content' }}
        />
      ) : (
        <Empty description="暂无可比房产数据" />
      )}
    </Card>
  );
};

export default PropertyComparables; 