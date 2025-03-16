import { useState, useEffect } from 'react';
import { 
  Typography, 
  Card, 
  Row, 
  Col, 
  Button, 
  Table, 
  message, 
  Popconfirm,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  Divider,
  Spin,
  Tabs,
  Upload,
  Radio,
  Alert
} from 'antd';
import { 
  CloudUploadOutlined, 
  FileAddOutlined, 
  CheckCircleOutlined,
  UploadOutlined,
  PlusOutlined,
  SettingOutlined
} from '@ant-design/icons';

const { Title, Paragraph, Text } = Typography;
const { Option } = Select;

const ModelManagement = () => {
  // 状态变量
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeKey, setActiveKey] = useState('existingModels');
  const [trainingForm] = Form.useForm();
  const [trainingVisible, setTrainingVisible] = useState(false);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [uploadVisible, setUploadVisible] = useState(false);
  const [fileList, setFileList] = useState([]);
  const [uploadForm] = Form.useForm();

  // 页面加载时获取模型列表
  useEffect(() => {
    fetchModels();
  }, []);

  // 获取模型列表
  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/models');
      const data = await response.json();
      if (response.ok) {
        setModels(data.models);
      } else {
        message.error('获取模型列表失败');
      }
    } catch (error) {
      console.error('获取模型列表错误:', error);
      message.error('获取模型列表失败');
    } finally {
      setLoading(false);
    }
  };

  // 启用模型
  const activateModel = async (modelId) => {
    try {
      const response = await fetch(`/api/models/${modelId}/activate`, {
        method: 'POST',
      });
      
      if (response.ok) {
        message.success('模型已成功激活');
        fetchModels(); // 刷新模型列表
      } else {
        const errorData = await response.json();
        message.error(`激活模型失败: ${errorData.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('激活模型错误:', error);
      message.error('激活模型失败，请稍后再试');
    }
  };

  // 删除模型
  const deleteModel = async (modelId) => {
    try {
      const response = await fetch(`/api/models/${modelId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        message.success('模型已成功删除');
        fetchModels(); // 刷新模型列表
      } else {
        const errorData = await response.json();
        message.error(`删除模型失败: ${errorData.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('删除模型错误:', error);
      message.error('删除模型失败，请稍后再试');
    }
  };

  // 开始模型训练
  const startTraining = async (values) => {
    try {
      setTrainingInProgress(true);
      setTrainingProgress(0);
      setTrainingLogs(['开始模型训练...']);
      
      // 模拟训练进度
      const interval = setInterval(() => {
        setTrainingProgress(prev => {
          const newProgress = prev + Math.floor(Math.random() * 10);
          if (newProgress >= 100) {
            clearInterval(interval);
            return 100;
          }
          return newProgress;
        });
        
        setTrainingLogs(prev => [
          ...prev, 
          `[${new Date().toLocaleTimeString()}] 训练进行中，处理数据批次 ${Math.floor(Math.random() * 1000)}...`
        ]);
      }, 2000);
      
      // 实际训练API调用
      const response = await fetch('/api/models/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });
      
      clearInterval(interval);
      
      if (response.ok) {
        setTrainingProgress(100);
        setTrainingLogs(prev => [...prev, '训练完成！']);
        message.success('模型训练成功');
        setTimeout(() => {
          setTrainingVisible(false);
          setTrainingInProgress(false);
          fetchModels(); // 刷新模型列表
        }, 2000);
      } else {
        const errorData = await response.json();
        message.error(`模型训练失败: ${errorData.message || '未知错误'}`);
        setTrainingInProgress(false);
      }
    } catch (error) {
      console.error('模型训练错误:', error);
      message.error('模型训练失败，请稍后再试');
      setTrainingInProgress(false);
    }
  };

  // 模型上传
  const uploadModel = async (values) => {
    if (fileList.length === 0) {
      message.error('请选择要上传的模型文件');
      return;
    }
    
    try {
      const formData = new FormData();
      formData.append('model_file', fileList[0].originFileObj);
      formData.append('name', values.name);
      formData.append('description', values.description);
      formData.append('model_type', values.model_type);
      
      const response = await fetch('/api/models/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        message.success('模型上传成功');
        setUploadVisible(false);
        uploadForm.resetFields();
        setFileList([]);
        fetchModels(); // 刷新模型列表
      } else {
        const errorData = await response.json();
        message.error(`模型上传失败: ${errorData.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('模型上传错误:', error);
      message.error('模型上传失败，请稍后再试');
    }
  };

  // 表格列定义
  const columns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '模型类型',
      dataIndex: 'model_type',
      key: 'model_type',
      render: (text) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (text) => new Date(text).toLocaleString('zh-CN'),
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (text) => `${(text * 100).toFixed(2)}%`,
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      render: (active) => (
        active ? 
        <Tag color="green" icon={<CheckCircleOutlined />}>当前使用中</Tag> : 
        <Tag color="default">未激活</Tag>
      ),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <div>
          {!record.is_active && (
            <Popconfirm
              title="确定要激活这个模型吗？"
              onConfirm={() => activateModel(record.id)}
              okText="确定"
              cancelText="取消"
            >
              <Button type="primary" size="small" style={{ marginRight: 8 }}>
                激活
              </Button>
            </Popconfirm>
          )}
          
          {!record.is_active && (
            <Popconfirm
              title="确定要删除这个模型吗？"
              onConfirm={() => deleteModel(record.id)}
              okText="确定"
              cancelText="取消"
            >
              <Button danger size="small">
                删除
              </Button>
            </Popconfirm>
          )}
        </div>
      ),
    },
  ];

  // 文件上传组件配置
  const uploadProps = {
    onRemove: () => {
      setFileList([]);
    },
    beforeUpload: (file) => {
      setFileList([file]);
      return false;
    },
    fileList,
  };

  // 模型列表显示
  const renderModelList = () => (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between' }}>
        <div>
          <Button 
            type="primary" 
            icon={<FileAddOutlined />} 
            onClick={() => setTrainingVisible(true)}
            style={{ marginRight: 8 }}
          >
            训练新模型
          </Button>
          
          <Button 
            icon={<UploadOutlined />} 
            onClick={() => setUploadVisible(true)}
          >
            上传预训练模型
          </Button>
        </div>
        
        <Button onClick={fetchModels}>刷新</Button>
      </div>
      
      <Table 
        columns={columns} 
        dataSource={models} 
        rowKey="id" 
        loading={loading}
        pagination={{ pageSize: 10 }}
      />
    </div>
  );

  // 训练新模型模态框
  const renderTrainingModal = () => (
    <Modal
      title="训练新模型"
      open={trainingVisible}
      onCancel={() => {
        if (!trainingInProgress) {
          setTrainingVisible(false);
          trainingForm.resetFields();
        }
      }}
      footer={trainingInProgress ? null : [
        <Button key="cancel" onClick={() => setTrainingVisible(false)}>
          取消
        </Button>,
        <Button 
          key="submit" 
          type="primary" 
          onClick={() => trainingForm.submit()}
        >
          开始训练
        </Button>,
      ]}
      width={700}
    >
      {trainingInProgress ? (
        <div>
          <div style={{ textAlign: 'center', marginBottom: 20 }}>
            <Spin />
            <div style={{ marginTop: 10 }}>
              <Text strong>模型训练中... {trainingProgress}%</Text>
            </div>
          </div>
          
          <div 
            style={{ 
              height: 300, 
              overflow: 'auto', 
              border: '1px solid #e8e8e8', 
              padding: 10, 
              backgroundColor: '#f5f5f5', 
              fontFamily: 'monospace' 
            }}
          >
            {trainingLogs.map((log, index) => (
              <div key={index}>{log}</div>
            ))}
          </div>
        </div>
      ) : (
        <Form
          form={trainingForm}
          layout="vertical"
          onFinish={startTraining}
          initialValues={{
            model_type: 'xgboost',
            dataset: 'default',
            test_size: 0.2,
            hyperparams: {
              n_estimators: 100,
              learning_rate: 0.1,
              max_depth: 6
            }
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="模型名称"
                rules={[{ required: true, message: '请输入模型名称' }]}
              >
                <Input placeholder="例如：XGBoost-2023-07" />
              </Form.Item>
            </Col>
            
            <Col span={12}>
              <Form.Item
                name="model_type"
                label="模型类型"
                rules={[{ required: true, message: '请选择模型类型' }]}
              >
                <Select>
                  <Option value="xgboost">XGBoost</Option>
                  <Option value="random_forest">随机森林</Option>
                  <Option value="linear_regression">线性回归</Option>
                  <Option value="neural_network">神经网络</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item
            name="description"
            label="模型描述"
          >
            <Input.TextArea rows={2} placeholder="输入模型的简要描述" />
          </Form.Item>
          
          <Divider orientation="left">训练参数</Divider>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="dataset"
                label="数据集"
                rules={[{ required: true, message: '请选择数据集' }]}
              >
                <Select>
                  <Option value="default">默认数据集</Option>
                  <Option value="extended">扩展数据集</Option>
                  <Option value="premium">高级数据集</Option>
                </Select>
              </Form.Item>
            </Col>
            
            <Col span={12}>
              <Form.Item
                name="test_size"
                label="测试集比例"
                rules={[{ required: true, message: '请选择测试集比例' }]}
              >
                <Select>
                  <Option value={0.1}>10%</Option>
                  <Option value={0.2}>20%</Option>
                  <Option value={0.3}>30%</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Divider orientation="left">高级参数</Divider>
          
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name={['hyperparams', 'n_estimators']}
                label="n_estimators"
                rules={[{ required: true, message: '请输入参数值' }]}
              >
                <Input type="number" />
              </Form.Item>
            </Col>
            
            <Col span={8}>
              <Form.Item
                name={['hyperparams', 'learning_rate']}
                label="learning_rate"
                rules={[{ required: true, message: '请输入参数值' }]}
              >
                <Input type="number" step={0.01} />
              </Form.Item>
            </Col>
            
            <Col span={8}>
              <Form.Item
                name={['hyperparams', 'max_depth']}
                label="max_depth"
                rules={[{ required: true, message: '请输入参数值' }]}
              >
                <Input type="number" />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      )}
    </Modal>
  );

  // 上传模型模态框
  const renderUploadModal = () => (
    <Modal
      title="上传预训练模型"
      open={uploadVisible}
      onCancel={() => {
        setUploadVisible(false);
        uploadForm.resetFields();
        setFileList([]);
      }}
      footer={[
        <Button key="cancel" onClick={() => setUploadVisible(false)}>
          取消
        </Button>,
        <Button 
          key="submit" 
          type="primary" 
          onClick={() => uploadForm.submit()}
        >
          上传
        </Button>,
      ]}
    >
      <Form
        form={uploadForm}
        layout="vertical"
        onFinish={uploadModel}
        initialValues={{
          model_type: 'xgboost',
        }}
      >
        <Form.Item
          name="name"
          label="模型名称"
          rules={[{ required: true, message: '请输入模型名称' }]}
        >
          <Input placeholder="例如：预训练XGBoost模型" />
        </Form.Item>
        
        <Form.Item
          name="model_type"
          label="模型类型"
          rules={[{ required: true, message: '请选择模型类型' }]}
        >
          <Select>
            <Option value="xgboost">XGBoost</Option>
            <Option value="random_forest">随机森林</Option>
            <Option value="linear_regression">线性回归</Option>
            <Option value="neural_network">神经网络</Option>
          </Select>
        </Form.Item>
        
        <Form.Item
          name="description"
          label="模型描述"
        >
          <Input.TextArea rows={2} placeholder="输入模型的简要描述" />
        </Form.Item>
        
        <Form.Item
          name="model_file"
          label="模型文件"
          rules={[{ required: true, message: '请上传模型文件' }]}
        >
          <Upload {...uploadProps} maxCount={1}>
            <Button icon={<UploadOutlined />}>选择文件</Button>
          </Upload>
        </Form.Item>
        
        <Alert
          message="支持的文件格式"
          description="支持.pkl, .joblib, .h5, .pb等模型文件格式。文件大小不超过50MB。"
          type="info"
          showIcon
        />
      </Form>
    </Modal>
  );

  return (
    <>
      <Title level={3}>模型管理</Title>
      <Paragraph>
        管理和训练用于房产估价的预测模型。您可以激活不同的模型，训练新的模型，或上传预训练的模型。
      </Paragraph>
      
      <Tabs 
        activeKey={activeKey} 
        onChange={setActiveKey}
        items={[
          {
            key: 'existingModels',
            label: '已有模型',
            children: renderModelList()
          }
        ]}
      />
      
      {renderTrainingModal()}
      {renderUploadModal()}
    </>
  );
};

export default ModelManagement; 