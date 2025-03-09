'use client';

import React from 'react';
import { Card, Typography, Row, Col, Tag, List, Timeline, Divider, Tooltip, Alert, Spin } from 'antd';
import { QuestionCircleOutlined, CheckCircleOutlined, CloseCircleOutlined, ExperimentOutlined, InfoCircleOutlined } from '@ant-design/icons';
import dynamic from 'next/dynamic';

// 使用动态导入并禁用 SSR
const Pie = dynamic(
  () => import('@ant-design/charts').then((mod) => mod.Pie),
  { ssr: false, loading: () => <div style={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Spin tip="图表加载中..." /></div> }
);

const { Paragraph, Title, Text } = Typography;

const PropertyModelExplanation = ({ modelExplanation, featureImportance }) => {
  if (!modelExplanation || !featureImportance || featureImportance.length === 0) {
    return null;
  }

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

  // 特征重要性饼图数据 - 使用实际影响值（如果有）或贡献值
  const generatePieData = () => {
    // 过滤掉SHAP验证信息对象
    const validFeatures = featureImportance.filter(f => f.feature !== '_SHAP_验证_');
    
    return validFeatures.slice(0, 5).map(item => {
      const value = Math.abs(item.contribution || item.importance * 100);
        
      return {
        type: translateFeatureName(item.feature),
        value: value,
        effect: item.effect
      };
    });
  };
  
  const pieData = generatePieData();
  
  const pieConfig = {
    data: pieData,
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name}: {percentage}',
    },
    // 添加颜色映射 - 正向特征使用绿色系，负向特征使用红色系
    color: ({ type, effect }) => {
      const positive = ['#389e0d', '#52c41a', '#73d13d', '#95de64', '#b7eb8f'];
      const negative = ['#cf1322', '#f5222d', '#ff4d4f', '#ff7875', '#ffa39e'];
      
      // 根据特征的索引和方向选择不同的颜色
      const typeIndex = pieData.findIndex(d => d.type === type);
      const palette = effect === 'positive' ? positive : negative;
      return palette[typeIndex % palette.length];
    },
    interactions: [
      {
        type: 'element-active',
      },
    ],
    tooltip: {
      formatter: (datum) => {
        const feature = featureImportance.find(f => translateFeatureName(f.feature) === datum.type);
        return {
          name: datum.type,
          value: `¥${Math.abs(feature.contribution || 0).toLocaleString()}`,
        };
      },
    },
  };

  // 查找是否有详细的特征解释
  const hasDetailedExplanations = modelExplanation.feature_explanations || modelExplanation.top_features_detail;
  const explanations = modelExplanation.feature_explanations || [];
  const explanationSummary = modelExplanation.explanation_summary || '';
  
  // 计算积极和消极特征的总贡献
  const positiveContribution = modelExplanation.positive_contribution || 0;
  const negativeContribution = modelExplanation.negative_contribution || 0;
  const positivePercent = modelExplanation.positive_contribution_percent || 0;
  const negativePercent = modelExplanation.negative_contribution_percent || 0;

  return (
    <Card title="AI模型解释详情" style={{ marginTop: '24px' }} bordered={false}>
      <Row gutter={[24, 24]}>
        <Col xs={24} md={12}>
          <Title level={5}>
            模型技术详情
            <Tooltip title="这里展示了用于预测该房产价格的AI模型的技术细节">
              <QuestionCircleOutlined style={{ marginLeft: '8px', fontSize: '16px' }} />
            </Tooltip>
          </Title>
          
          <List
            size="small"
            bordered
            dataSource={[
              { label: '模型类型', value: modelExplanation.model_type },
              { label: 'R²决定系数', value: modelExplanation.r2_score },
              { label: '平均绝对误差', value: `A$${(modelExplanation.mae / 10000).toFixed(1)}万` },
              { label: '平均绝对百分比误差', value: `${modelExplanation.mape}%` },
              { label: '使用特征数量', value: modelExplanation.feature_count },
              { label: '预测置信度', value: `${modelExplanation.prediction_confidence}%` }
            ]}
            renderItem={item => (
              <List.Item>
                <Text strong>{item.label}：</Text> {item.value}
              </List.Item>
            )}
          />
          
          <Divider />
          
          {/* 特征贡献占比分析 */}
          {(positiveContribution > 0 || negativeContribution < 0) && (
            <>
              <Title level={5}>特征贡献占比</Title>
              <Row gutter={16}>
                <Col span={12}>
                  <Paragraph>
                    <Text type="success">正向贡献: </Text>
                    <Text strong>A${positiveContribution.toLocaleString()}</Text>
                    <br />
                    <small>（占总价的 {positivePercent}%）</small>
                  </Paragraph>
                </Col>
                <Col span={12}>
                  <Paragraph>
                    <Text type="danger">负向影响: </Text>
                    <Text strong>A${Math.abs(negativeContribution).toLocaleString()}</Text>
                    <br />
                    <small>（占总价的 {negativePercent}%）</small>
                  </Paragraph>
                </Col>
              </Row>
              <Divider />
            </>
          )}
          
          <Title level={5}>价格预测过程解释</Title>
          <Timeline
            items={[
              {
                dot: <ExperimentOutlined style={{ fontSize: '16px' }} />,
                color: 'blue',
                children: (
                  <Text>AI模型分析了房产的{modelExplanation.feature_count}个特征数据</Text>
                ),
              },
              {
                dot: <CheckCircleOutlined style={{ fontSize: '16px' }} />,
                color: 'green',
                children: (
                  <>
                    <Text>模型识别出最重要的正向影响因素：</Text>
                    <div style={{ marginTop: '4px' }}>
                      {modelExplanation.top_positive_features.map((feature, index) => (
                        <Tag color="green" key={index}>{translateFeatureName(feature)}</Tag>
                      ))}
                    </div>
                  </>
                ),
              },
              {
                dot: <CloseCircleOutlined style={{ fontSize: '16px' }} />,
                color: 'red',
                children: (
                  <>
                    <Text>模型识别出最重要的负向影响因素：</Text>
                    <div style={{ marginTop: '4px' }}>
                      {modelExplanation.top_negative_features.map((feature, index) => (
                        <Tag color="red" key={index}>{translateFeatureName(feature)}</Tag>
                      ))}
                    </div>
                  </>
                ),
              },
              {
                color: 'gray',
                children: (
                  <Text>
                    最终，模型基于{modelExplanation.r2_score}的R²决定系数和{modelExplanation.mape}%的平均误差率，
                    得出了{modelExplanation.prediction_confidence}%的置信度预测。
                  </Text>
                ),
              },
            ]}
          />
        </Col>
        
        <Col xs={24} md={12}>
          <Title level={5}>特征重要性分布</Title>
          <Paragraph>
            下图展示了对该房产价格影响最大的前5个因素及其相对重要性：
          </Paragraph>
          <div style={{ height: '300px' }}>
            <Pie {...pieConfig} />
          </div>
          
          <Divider />
          
          {/* 详细特征解释 */}
          {explanationSummary && (
            <Alert
              message="AI分析总结"
              description={explanationSummary}
              type="info"
              showIcon
              icon={<InfoCircleOutlined />}
              style={{ marginBottom: '16px' }}
            />
          )}
          
          {hasDetailedExplanations && (
            <>
              <Title level={5}>详细特征影响解释</Title>
              <List
                size="small"
                bordered
                dataSource={explanations}
                renderItem={(item, index) => (
                  <List.Item>
                    <Text>{index + 1}. {item}</Text>
                  </List.Item>
                )}
              />
            </>
          )}
          
          {!hasDetailedExplanations && (
            <Paragraph>
              <Text strong>模型解读：</Text> 根据AI分析，该房产的价格主要受
              <Tag color="blue">{translateFeatureName(featureImportance[0]?.feature)}</Tag>
              和
              <Tag color="blue">{translateFeatureName(featureImportance[1]?.feature)}</Tag>
              的影响。特别是
              <Tag color="green">{translateFeatureName(modelExplanation.top_positive_features[0])}</Tag>
              对价格有显著提升作用，而
              <Tag color="red">{translateFeatureName(modelExplanation.top_negative_features[0])}</Tag>
              则略有拖累。
            </Paragraph>
          )}
          
          <Paragraph>
            此预测基于<Text code>{modelExplanation.model_type}</Text>模型，该模型在训练过程中表现出
            <Text strong>{modelExplanation.r2_score}</Text>的R²决定系数，意味着模型可以解释约
            <Text strong>{(modelExplanation.r2_score * 100).toFixed(0)}%</Text>的价格变化。
          </Paragraph>
        </Col>
      </Row>
    </Card>
  );
};

export default PropertyModelExplanation; 