'use client';

import React from 'react';
import { Card, Typography, Empty, Spin } from 'antd';
import dynamic from 'next/dynamic';

// 使用动态导入并禁用 SSR
const Line = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Line),
  { ssr: false, loading: () => <div style={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Text, Paragraph } = Typography;

const PropertyPriceTrends = ({ priceTrends }) => {
  if (!priceTrends || priceTrends.length === 0) {
    return (
      <Card title="房产价格趋势" style={{ marginTop: '24px' }} bordered={false}>
        <Empty description="暂无价格趋势数据" />
      </Card>
    );
  }
  
  const config = {
    data: priceTrends,
    xField: 'date',
    yField: 'price',
    point: {
      size: 5,
      shape: 'diamond',
    },
    label: {
      style: {
        fill: '#aaa',
      },
    },
    yAxis: {
      title: {
        text: '价格（元）',
      },
      label: {
        formatter: (v) => `A$${(v / 10000).toFixed(0)}万`,
      },
    },
    tooltip: {
      formatter: (data) => {
        return { name: '估价', value: `A$${(data.price / 10000).toFixed(2)}万` };
      },
    },
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
    color: '#1890ff',
  };

  return (
    <Card title="房产价格趋势" style={{ marginTop: '24px' }} bordered={false}>
      <Paragraph>
        该图表展示了该房产近两年的估价趋势变化，反映了该区域房产市场的波动情况。
      </Paragraph>
      <div style={{ height: '350px' }}>
        <Line {...config} />
      </div>
    </Card>
  );
};

export default PropertyPriceTrends; 