import React from 'react';
import { Card, Statistic, Divider } from 'antd';
import { 
  DollarOutlined, 
  ArrowUpOutlined, 
  ArrowDownOutlined 
} from '@ant-design/icons';

const PropertyPriceResult = ({ property }) => {
  if (!property) return null;
  
  return (
    <Card title="估价结果" bordered={false} style={{ marginBottom: '24px' }}>
      <Statistic
        title="AI估价"
        value={property.predicted_price}
        precision={0}
        formatter={value => `A$${(value / 10000).toFixed(0)}万`}
        valueStyle={{ color: '#1890ff', fontSize: '24px' }}
        prefix={<DollarOutlined />}
      />
      
      {property.actual_price > 0 && (
        <>
          <Divider style={{ margin: '16px 0' }} />
          
          <Statistic
            title="实际价格"
            value={property.actual_price}
            precision={0}
            formatter={value => `A$${(value / 10000).toFixed(0)}万`}
            valueStyle={{ fontSize: '18px' }}
          />
          
          <Statistic
            title="误差百分比"
            value={property.error_percent}
            precision={1}
            suffix="%"
            valueStyle={{ 
              color: Math.abs(property.error_percent) < 3 ? '#3f8600' : 
                    Math.abs(property.error_percent) < 5 ? '#faad14' : '#cf1322',
              fontSize: '18px'
            }}
            prefix={property.error_percent > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
          />
        </>
      )}
      
      <Divider style={{ margin: '16px 0' }} />
      
      {property.area > 0 && (
        <div style={{ marginTop: '16px' }}>
          <Statistic
            title="单价"
            value={(property.predicted_price / property.area).toFixed(0)}
            suffix="元/m²"
            valueStyle={{ fontSize: '16px' }}
          />
        </div>
      )}
    </Card>
  );
};

export default PropertyPriceResult; 