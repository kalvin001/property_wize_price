import React from 'react';
import { Card, Descriptions, Tag, Divider, Empty, Typography } from 'antd';
import { ClockCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;

const PropertyBasicInfo = ({ property }) => {
  if (!property) return null;

  return (
    <Card title="房产信息" bordered={false} style={{ marginBottom: '24px' }}>
      <Descriptions column={1} bordered>
        <Descriptions.Item label="报告ID">{property.id}</Descriptions.Item>
        <Descriptions.Item label="房产类型">
          <Tag color={
            property.type && property.type.includes('公寓') ? 'blue' : 
            property.type && property.type.includes('别墅') ? 'green' : 'purple'
          }>
            {property.type || '未知类型'}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="所在区域">{property.region}</Descriptions.Item>
        <Descriptions.Item label="小区名称">{property.community}</Descriptions.Item>
        <Descriptions.Item label="面积">
          {property.area ? `${property.area}m²` : '未知'}
        </Descriptions.Item>
        <Descriptions.Item label="户型">
          {property.beds > 0 ? `${property.beds}室${property.baths}卫` : '未知'}
        </Descriptions.Item>
        {property.floor > 0 && (
          <Descriptions.Item label="楼层">
            {property.floor}/{property.total_floors}层
          </Descriptions.Item>
        )}
        <Descriptions.Item label="朝向">{property.orientation}</Descriptions.Item>
        <Descriptions.Item label="装修">{property.decoration}</Descriptions.Item>
        {property.year_built > 0 && (
          <Descriptions.Item label="建成年份">{property.year_built}年</Descriptions.Item>
        )}
        <Descriptions.Item label="更新日期">
          <ClockCircleOutlined style={{ marginRight: '8px' }} />
          {property.updated_at}
        </Descriptions.Item>
        {property.postcode && (
          <Descriptions.Item label="邮编">{property.postcode}</Descriptions.Item>
        )}
        {property.carpark > 0 && (
          <Descriptions.Item label="车位">{property.carpark}个</Descriptions.Item>
        )}
      </Descriptions>

      <Divider orientation="left">房产特点</Divider>
      <div>
        {property.key_features && property.key_features.length > 0 ? (
          property.key_features.map((feature, index) => (
            <Tag key={index} color="blue" style={{ margin: '0 8px 8px 0' }}>{feature}</Tag>
          ))
        ) : (
          <Empty description="暂无特点数据" />
        )}
      </div>
    </Card>
  );
};

export default PropertyBasicInfo; 