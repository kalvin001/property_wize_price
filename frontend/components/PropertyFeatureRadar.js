'use client';

import React from 'react';
import { Card, Typography, Empty, Row, Col, Spin } from 'antd';
import dynamic from 'next/dynamic';

// 使用动态导入并禁用 SSR
const Radar = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Radar),
  { ssr: false, loading: () => <div style={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Paragraph, Text } = Typography;

const PropertyFeatureRadar = ({ property, featureImportance }) => {
  if (!property || !property.features || !featureImportance || featureImportance.length === 0) {
    return (
      <Card title="房产特征雷达图" style={{ marginTop: '24px' }} bordered={false}>
        <Empty description="暂无房产特征数据" />
      </Card>
    );
  }

  // 整理数据，将特征值标准化为0-1之间
  const normalizeFeatureValue = (feature, value) => {
    // 针对不同特征设置不同的标准化方法
    const normalizers = {
      'prop_area': () => Math.min(value / 200, 1),  // 假设最大200平方米
      'prop_bed': () => Math.min(value / 5, 1),     // 假设最大5间卧室
      'prop_bath': () => Math.min(value / 3, 1),    // 假设最大3间浴室
      'prop_age': () => 1 - Math.min(value / 50, 1), // 假设最大50年，越新越好
      'land_size': () => Math.min(value / 1000, 1),  // 假设最大1000平方米
    };
    
    return normalizers[feature] ? normalizers[feature]() : 0.5;
  };

  // 生成雷达图数据
  const generateRadarData = () => {
    const importantFeatures = featureImportance.slice(0, 5); // 取前5个重要特征
    
    const data = [];
    importantFeatures.forEach(featureInfo => {
      const featureName = featureInfo.feature;
      const featureValue = property.features[featureName];
      
      if (featureValue !== undefined) {
        // 为每个特征生成两条数据：当前房产和平均水平
        data.push({
          item: translateFeatureName(featureName),
          score: normalizeFeatureValue(featureName, featureValue),
          type: '当前房产'
        });
        
        data.push({
          item: translateFeatureName(featureName),
          score: 0.5, // 假设平均值为0.5
          type: '市场平均'
        });
      }
    });
    
    return data;
  };

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

  const radarData = generateRadarData();
  
  const config = {
    data: radarData,
    xField: 'item',
    yField: 'score',
    seriesField: 'type',
    meta: {
      score: {
        alias: '分数',
        min: 0,
        max: 1,
      },
    },
    xAxis: {
      line: null,
      tickLine: null,
    },
    yAxis: {
      label: false,
      grid: {
        alternateColor: 'rgba(0, 0, 0, 0.04)',
      },
    },
    point: {
      size: 3,
    },
    area: {
      style: {
        fillOpacity: 0.3,
      }
    },
    legend: {
      position: 'bottom',
    },
  };

  return (
    <Card title="房产特征雷达图" style={{ marginTop: '24px' }} bordered={false}>
      <Paragraph>
        该雷达图展示了房产的关键特征与市场平均水平的对比，帮助您理解该房产的优势和劣势。
      </Paragraph>
      <div style={{ height: '350px' }}>
        <Radar {...config} />
      </div>
    </Card>
  );
};

export default PropertyFeatureRadar; 