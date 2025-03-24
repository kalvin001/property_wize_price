import { useState, useEffect } from 'react';
import { 
  Typography, 
  Card, 
  Row, 
  Col, 
  Button, 
  message, 
  Spin, 
  Divider,
  Alert,
  Space
} from 'antd';
import { PlusOutlined, SyncOutlined } from '@ant-design/icons';
import ModelCard from '../components/ModelCard';

const { Title, Paragraph } = Typography;

const ModelManagement = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeModel, setActiveModel] = useState(null);
  const [activating, setActivating] = useState(false);

  // 获取所有模型
  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models);
        
        // 获取当前激活的模型
        await fetchActiveModel();
      } else {
        message.error('获取模型列表失败');
      }
    } catch (error) {
      console.error('获取模型列表出错:', error);
      message.error('获取模型列表出错');
    } finally {
      setLoading(false);
    }
  };

  // 获取当前激活的模型
  const fetchActiveModel = async () => {
    try {
      const response = await fetch('/api/model/info');
      if (response.ok) {
        const data = await response.json();
        
        if (data.model_loaded) {
          setActiveModel(data.model_path || data.model_name);
        } else {
          setActiveModel(null);
        }
      }
    } catch (error) {
      console.error('获取激活模型信息出错:', error);
    }
  };

  // 初始化加载
  useEffect(() => {
    fetchModels();
  }, []);

  // 激活模型
  const handleActivateModel = async (model) => {
    if (activating) return;
    
    setActivating(true);
    try {
      const modelPath = model.path.split('/').pop();
      const response = await fetch(`/api/models/${modelPath}/activate`, {
        method: 'POST'
      });
      
      if (response.ok) {
        message.success(`模型 ${model.model_type || model.name || modelPath} 已激活`);
        setActiveModel(model.path);
        
        // 刷新列表以确保状态一致
        await fetchModels();
      } else {
        const errorData = await response.json();
        message.error(`激活模型失败: ${errorData.detail || '未知错误'}`);
      }
    } catch (error) {
      console.error('激活模型出错:', error);
      message.error('激活模型出错');
    } finally {
      setActivating(false);
    }
  };

  // 创建新模型
  const handleCreateModel = () => {
    // 这里可以打开一个模态框或者导航到训练页面
    message.info('目前尚未实现模型训练功能，请敬请期待');
  };

  return (
    <div className="model-management-container">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={2}>模型管理</Title>
        <Space>
          <Button 
            type="primary" 
            icon={<SyncOutlined />}
            onClick={fetchModels} 
            loading={loading}
          >
            刷新
          </Button>
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={handleCreateModel}
          >
            创建新模型
          </Button>
        </Space>
      </div>
      
      <Paragraph>
        在此页面可以管理所有可用的房产估价模型。您可以查看各模型的评估指标、激活所需模型用于预测，或者训练新的模型。
      </Paragraph>
      
      {activeModel && (
        <Alert
          message="当前激活模型"
          description={`模型路径: ${activeModel}`}
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}
      
      <Divider>可用模型</Divider>
      
      <Spin spinning={loading}>
        {models.length > 0 ? (
          <Row gutter={[16, 16]}>
            {models.map((model, index) => (
              <Col xs={24} sm={24} md={12} lg={8} key={index}>
                <ModelCard 
                  model={model}
                  onActivate={() => handleActivateModel(model)}
                  isActive={activeModel && model.path && model.path.includes(activeModel)}
                  showDetails={true}
                />
              </Col>
            ))}
          </Row>
        ) : (
          <Card>
            <div style={{ textAlign: 'center', padding: '24px 0' }}>
              <Typography.Text>暂无可用模型</Typography.Text>
            </div>
          </Card>
        )}
      </Spin>
      
      <Divider>如何使用</Divider>
      
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="使用指南" bordered={false}>
            <Paragraph>
              <strong>激活模型</strong>: 点击模型卡片上的"激活模型"按钮，将该模型设置为当前使用的预测模型。
            </Paragraph>
            <Paragraph>
              <strong>查看详情</strong>: 点击"详情"按钮可以查看模型的详细评估指标和特征重要性。
            </Paragraph>
            <Paragraph>
              <strong>创建新模型</strong>: 点击"创建新模型"按钮开始训练新的估价模型（功能开发中）。
            </Paragraph>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ModelManagement; 