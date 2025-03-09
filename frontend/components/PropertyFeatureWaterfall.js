'use client';

import React from 'react';
import { Card, Typography, Empty, Alert, Spin } from 'antd';
import dynamic from 'next/dynamic';

// 使用动态导入并禁用 SSR
const Waterfall = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Waterfall),
  { ssr: false, loading: () => <div style={{ height: '350px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Paragraph, Title, Text } = Typography;

const PropertyFeatureWaterfall = ({ predPrice, featureImportance }) => {
  if (!featureImportance || featureImportance.length === 0 || !predPrice) {
    return (
      <Card title="特征贡献瀑布图" style={{ marginTop: '24px' }} bordered={false}>
        <Empty description="暂无特征贡献数据" />
      </Card>
    );
  }

  // 过滤掉SHAP验证信息对象
  const validFeatures = featureImportance.filter(f => f.feature !== '_SHAP_验证_');
  
  // 获取SHAP验证信息，如果存在的话
  const shapValidation = featureImportance.find(f => f.feature === '_SHAP_验证_');
  
  // 特征名称翻译
  const translateFeatureName = (name) => {
    const translations = {
      'prop_area': '建筑面积',
      'prop_bed': '卧室数量',
      'prop_bath': '浴室数量',
      'prop_age': '房屋年龄',
      'land_size': '土地面积',
      'garage_spaces': '车库数量',
      'num_schools': '学校数量',
      'distance_cbd': '距市中心',
      'distance_train': '距火车站',
      'distance_beach': '距海滩',
    };
    
    return translations[name] || name;
  };

  // 为瀑布图准备数据
  const generateWaterfallData = () => {
    // 按贡献绝对值大小排序，取前5个特征
    const topFeatures = [...validFeatures]
      .sort((a, b) => Math.abs(b.contribution || 0) - Math.abs(a.contribution || 0))
      .slice(0, 5);
    
    // 基础价格 - 使用SHAP验证中的base_value或者根据贡献计算
    const basePrice = shapValidation ? 
      shapValidation.base_value : 
      predPrice - topFeatures.reduce((sum, feature) => sum + (feature.contribution || 0), 0);
    
    // 创建瀑布图数据
    const data = [
      { type: '基准价格', value: basePrice, isBase: true },
    ];
    
    // 添加各个特征的贡献
    topFeatures.forEach(feature => {
      const impact = feature.contribution || 0;
      data.push({
        type: translateFeatureName(feature.feature),
        value: impact,
        effect: feature.effect
      });
    });
    
    // 添加最终价格
    data.push({ type: '最终价格', value: predPrice, isTotal: true });
    
    return data;
  };
  
  const waterfallData = generateWaterfallData();
  
  const waterfallConfig = {
    data: waterfallData,
    xField: 'type',
    yField: 'value',
    seriesField: 'type',
    formatter: (datum) => {
      if (datum.isBase || datum.isTotal) {
        return { value: `¥${Math.round(datum.value).toLocaleString()}` };
      }
      return { 
        value: `${datum.value >= 0 ? '+' : ''}¥${Math.round(datum.value).toLocaleString()}` 
      };
    },
    color: ({ type, effect, isBase, isTotal }) => {
      if (isBase) return '#5B8FF9';
      if (isTotal) return '#5AD8A6';
      return effect === 'positive' ? '#73d13d' : '#ff7875';
    },
    label: {
      style: { fontSize: 12 },
      position: 'top',
      layout: [
        { type: 'adjust-color' }
      ],
    },
    waterfallStyle: ({ type, effect, isBase, isTotal }) => {
      const style = { 
        fill: isBase ? '#5B8FF9' : isTotal ? '#5AD8A6' : effect === 'positive' ? '#73d13d' : '#ff7875',
        stroke: isBase ? '#5B8FF9' : isTotal ? '#5AD8A6' : effect === 'positive' ? '#52c41a' : '#f5222d',
        lineWidth: 2
      };
      return style;
    },
    risingFill: '#73d13d',
    fallingFill: '#ff7875',
    yAxis: {
      label: {
        formatter: (v) => `¥${Number(v).toLocaleString()}`,
      },
    },
    tooltip: {
      formatter: (datum) => {
        if (datum.isBase) {
          return { name: '基准价格', value: `¥${Math.round(datum.value).toLocaleString()}` };
        }
        if (datum.isTotal) {
          return { name: '最终价格', value: `¥${Math.round(datum.value).toLocaleString()}` };
        }
        return { 
          name: datum.type, 
          value: `${datum.value >= 0 ? '+' : ''}¥${Math.round(datum.value).toLocaleString()}` 
        };
      },
    },
  };

  return (
    <Card title="特征贡献瀑布图" style={{ marginTop: '24px' }} bordered={false}>
      {shapValidation && shapValidation.is_valid === false && (
        <Alert
          message="SHAP验证提示"
          description="当前SHAP值计算有一定偏差，瀑布图可能不能完全反映价格形成过程。"
          type="warning"
          showIcon
          style={{ marginBottom: '16px' }}
        />
      )}
      <Paragraph>
        下图直观展示了各主要特征对房产价格的贡献，从基准价格到最终预测价格的变化过程：
      </Paragraph>
      <div style={{ height: '350px' }}>
        <Waterfall {...waterfallConfig} />
      </div>
      <Paragraph style={{ marginTop: '16px', fontSize: '12px', color: '#888' }}>
        注：绿色表示正向影响（提升价格），红色表示负向影响（降低价格）。基准价格是由SHAP计算得出的基础价值，各特征贡献共同构成最终价格。
      </Paragraph>
    </Card>
  );
};

export default PropertyFeatureWaterfall; 