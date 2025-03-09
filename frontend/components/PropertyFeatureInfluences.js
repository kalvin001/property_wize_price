import React, { useState } from 'react';
import { Card, Table, Empty, Typography, Progress, Tag, Row, Col, Divider, Tooltip, Statistic, Alert } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;

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

const PropertyFeatureInfluences = ({ featureImportance }) => {
  // 确保有数据
  if (!featureImportance || featureImportance.length === 0) {
    return (
      <Card title="价格影响因素分析" style={{ marginTop: '24px' }} bordered={false}>
        <Empty description="暂无特征重要性数据" />
      </Card>
    );
  }

  // 过滤掉SHAP验证信息对象
  const validFeatures = featureImportance.filter(f => f.feature !== '_SHAP_验证_');
  
  // 获取SHAP验证信息，如果存在的话
  const shapValidation = featureImportance.find(f => f.feature === '_SHAP_验证_');
  
  // 将数据分为正向和负向影响
  const positiveFeatures = validFeatures.filter(f => f.effect === 'positive');
  const negativeFeatures = validFeatures.filter(f => f.effect === 'negative');

  // 表格列定义 - 增强版
  const featureColumns = [
    {
      title: '特征',
      dataIndex: 'feature',
      key: 'feature',
      render: (text) => <span>{translateFeatureName(text)}</span>,
    },
    {
      title: '影响方向',
      dataIndex: 'effect',
      key: 'effect',
      render: (effect, record) => (
        <Tag color={effect === 'positive' ? '#3f8600' : '#cf1322'}>
          {effect === 'positive' ? '正向影响' : '负向影响'}
          {record.value_direction && 
            <Tooltip title={`该特征值${record.value_direction}`}>
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          }
        </Tag>
      ),
    },
    {
      title: (
        <span>
          特征价值贡献
          <Tooltip title="该特征对总价的贡献值">
            <QuestionCircleOutlined style={{ marginLeft: '4px' }} />
          </Tooltip>
        </span>
      ),
      dataIndex: 'contribution',
      key: 'contribution',
      render: (value, record) => (
        <>
          <Text type={record.effect === 'positive' ? 'success' : 'danger'}>
            {record.effect === 'positive' ? '+' : '-'}A${Math.abs(value).toLocaleString()}
          </Text>
          <div>
            <small>({record.contribution_percent}%)</small>
          </div>
        </>
      ),
    },
    {
      title: (
        <span>
          相对基准变化
          <Tooltip title="特征相对于基准值的变化比例">
            <QuestionCircleOutlined style={{ marginLeft: '4px' }} />
          </Tooltip>
        </span>
      ),
      dataIndex: 'relative_change',
      key: 'relative_change',
      render: (value, record) => (
        <>
          <Text type={record.effect === 'positive' ? 'success' : 'danger'}>
            {record.effect === 'positive' ? '+' : '-'}{Math.abs(value).toFixed(2)}%
          </Text>
        </>
      ),
    },
    {
      title: (
        <span>
          重要性
          <Tooltip title="特征在预测中的重要程度">
            <QuestionCircleOutlined style={{ marginLeft: '4px' }} />
          </Tooltip>
        </span>
      ),
      dataIndex: 'importance',
      key: 'importance',
      render: (value, record) => {
        // 计算适当的百分比值用于进度条显示
        const maxImportance = Math.max(...validFeatures.map(f => f.importance || 0));
        const percentValue = maxImportance > 0 ? (value / maxImportance * 100).toFixed(1) : 0;
        
        return (
          <Progress 
            percent={percentValue} 
            size="small"
            format={(percent) => `${percent}%`}
            status={record.effect === 'positive' ? 'success' : 'exception'}
            strokeColor={record.effect === 'positive' ? '#3f8600' : '#cf1322'}
          />
        );
      },
    },
  ];

  // 添加方向描述数据
  const enhancedData = validFeatures.map(item => {
    let featureDescription = '';
    
    if (item.value_direction) {
      if (item.effect === 'positive' && item.value_direction === '高于平均') {
        featureDescription = `${translateFeatureName(item.feature)}高于平均水平，对价格有正向提升作用`;
      } else if (item.effect === 'positive' && item.value_direction === '低于平均') {
        featureDescription = `尽管${translateFeatureName(item.feature)}低于平均水平，但仍对价格有正向作用`;
      } else if (item.effect === 'negative' && item.value_direction === '高于平均') {
        featureDescription = `${translateFeatureName(item.feature)}高于平均水平，但对价格有负向影响`;
      } else if (item.effect === 'negative' && item.value_direction === '低于平均') {
        featureDescription = `${translateFeatureName(item.feature)}低于平均水平，对价格有负向影响`;
      } else {
        featureDescription = `${translateFeatureName(item.feature)}对价格的${item.effect === 'positive' ? '正向' : '负向'}影响`;
      }
    } else {
      featureDescription = `${translateFeatureName(item.feature)}对价格的${item.effect === 'positive' ? '正向' : '负向'}影响`;
    }
    
    return {
      ...item,
      description: featureDescription
    };
  });

  // 计算总影响
  const totalPositive = positiveFeatures.reduce((sum, feature) => sum + Math.abs(feature.contribution), 0);
  const totalNegative = negativeFeatures.reduce((sum, feature) => sum + Math.abs(feature.contribution), 0);
  const topPositive = positiveFeatures.length > 0 ? positiveFeatures[0] : null;
  const topNegative = negativeFeatures.length > 0 ? negativeFeatures[0] : null;

  // 直观展示特征影响 - 使用简单的进度条
  const SimpleFeatureBar = ({ feature, value, maxValue, effect }) => {
    const percent = Math.min(100, Math.abs(value) / maxValue * 100);
    const barColor = effect === 'positive' ? '#3f8600' : '#cf1322';
    return (
      <div style={{ marginBottom: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <Text>{translateFeatureName(feature)}</Text>
          <Text type={effect === 'positive' ? 'success' : 'danger'}>
            {effect === 'positive' ? '+' : '-'}A${Math.abs(value).toLocaleString()}
          </Text>
        </div>
        <div style={{ height: '12px', background: '#f0f0f0', borderRadius: '6px', overflow: 'hidden' }}>
          <div 
            style={{ 
              height: '100%', 
              width: `${percent}%`, 
              background: barColor,
              borderRadius: '6px'
            }} 
          />
        </div>
      </div>
    );
  };

  // 取前10个最重要特征用于展示条形图
  const topFeatures = [...validFeatures]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10);
  
  const maxValue = Math.max(...topFeatures.map(f => Math.abs(f.contribution)));

  return (
    <Card 
      title="价格影响因素分析 (SHAP值解释)" 
      style={{ marginTop: '24px' }} 
      bordered={false}
      extra={
        shapValidation && (
          <Tooltip title={`SHAP基准值: ${shapValidation.base_value.toLocaleString()}元，验证是否有效: ${shapValidation.is_valid ? '是' : '否'}`}>
            <Tag color={shapValidation.is_valid ? 'green' : 'orange'}>
              {shapValidation.is_valid ? 'SHAP验证有效' : 'SHAP验证偏差'}
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tag>
          </Tooltip>
        )
      }
    >
      {shapValidation && (
        <Row gutter={[16, 16]} style={{ marginBottom: '20px' }}>
          <Col span={24}>
            <Alert
              message="SHAP基准价值与贡献"
              description={
                <>
                  <p>基准价值: <Text strong>A${shapValidation.base_value.toLocaleString()}</Text></p>
                  <p>特征总贡献: <Text strong>{shapValidation.total_contribution >= 0 ? '+' : '-'}A${Math.abs(shapValidation.total_contribution).toLocaleString()}</Text></p>
                  <p>计算后价格: <Text strong>A${shapValidation.calculated_price.toLocaleString()}</Text></p>
                  <p>实际预测价格: <Text strong>A${shapValidation.model_prediction.toLocaleString()}</Text></p>
                  <p>差异比例: <Text type={Math.abs(shapValidation.price_diff_percent) < 1 ? 'success' : 'warning'}>{shapValidation.price_diff_percent}%</Text></p>
                </>
              }
              type="info"
              showIcon
            />
          </Col>
        </Row>
      )}
      <Row gutter={[16, 16]}>
        {/* 总体统计信息 */}
        <Col span={24}>
          <Row gutter={16}>
            <Col span={12}>
              <Statistic
                title="正向因素总影响"
                value={totalPositive}
                precision={0}
                valueStyle={{ color: '#3f8600' }}
                prefix={<ArrowUpOutlined />}
                suffix="元"
              />
              {topPositive && (
                <div style={{ marginTop: '8px' }}>
                  <Text type="secondary">最大提升：</Text>
                  <Tag color="green">{translateFeatureName(topPositive.feature)}</Tag>
                </div>
              )}
            </Col>
            <Col span={12}>
              <Statistic
                title="负向因素总影响"
                value={Math.abs(totalNegative)}
                precision={0}
                valueStyle={{ color: '#cf1322' }}
                prefix={<ArrowDownOutlined />}
                suffix="元"
              />
              {topNegative && (
                <div style={{ marginTop: '8px' }}>
                  <Text type="secondary">最大降低：</Text>
                  <Tag color="red">{translateFeatureName(topNegative.feature)}</Tag>
                </div>
              )}
            </Col>
          </Row>
          <Divider />
        </Col>

        {/* 简单的可视化展示 */}
        <Col span={24} style={{ marginBottom: '20px' }}>
          <div style={{ marginBottom: '10px' }}>
            <Typography.Title level={5}>主要特征影响</Typography.Title>
          </div>
          {topFeatures.map((feature, index) => (
            <SimpleFeatureBar 
              key={index}
              feature={feature.feature}
              value={feature.contribution}
              maxValue={maxValue}
              effect={feature.effect}
            />
          ))}
          <Divider />
        </Col>

        {/* 特征详情表格 */}
        <Col span={24}>
          <Table 
            columns={featureColumns} 
            dataSource={enhancedData}
            rowKey="feature"
            pagination={false}
            size="middle"
          />
          <Text type="secondary" style={{ marginTop: '16px', display: 'block' }}>
            注：重要性基于SHAP值计算，表示各特征对最终价格的影响程度。
          </Text>
        </Col>
      </Row>
    </Card>
  );
};

export default PropertyFeatureInfluences; 