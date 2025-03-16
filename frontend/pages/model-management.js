import { useState, useEffect } from 'react';
import Head from 'next/head';
import { 
  Layout, 
  Typography, 
  Card, 
  Row, 
  Col, 
  Table,
  Tag,
  Button,
  Divider,
  Spin,
  Alert,
  Modal,
  Form,
  Select,
  InputNumber,
  notification,
  Tabs,
  Space,
  Drawer,
  Descriptions,
  Badge,
  Popconfirm
} from 'antd';
import { 
  PlusOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  SettingOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import Link from 'next/link';
import moment from 'moment';
import MainHeader from '../components/Header';

const { Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

export default function ModelManagement() {
  // 状态管理
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [trainModalVisible, setTrainModalVisible] = useState(false);
  const [trainLoading, setTrainLoading] = useState(false);
  const [activeModel, setActiveModel] = useState(null);
  const [modelDetailVisible, setModelDetailVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [paramModalVisible, setParamModalVisible] = useState(false);
  const [modelParams, setModelParams] = useState({});
  const [selectedModelType, setSelectedModelType] = useState('xgboost');
  const [modelTypes] = useState(['xgboost', 'linear', 'ridge', 'lasso', 'elasticnet']);
  const [modelForm] = Form.useForm();

  // 加载模型列表
  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/models');
      if (!response.ok) {
        throw new Error('获取模型列表失败');
      }
      const data = await response.json();
      setModels(data.models || []);
      
      // 获取当前激活的模型信息
      const infoResponse = await fetch('/api/model/info');
      if (infoResponse.ok) {
        const infoData = await infoResponse.json();
        if (infoData.model_path) {
          setActiveModel(infoData.model_path);
        }
      }
      
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // 初始加载
  useEffect(() => {
    fetchModels();
  }, []);

  // 获取模型参数
  const fetchModelParams = async (modelType) => {
    try {
      const response = await fetch(`/api/models/${modelType}/params`);
      if (!response.ok) {
        throw new Error(`获取${modelType}模型参数失败`);
      }
      const data = await response.json();
      setModelParams(data.params || {});
      
      // 设置表单默认值
      const initialValues = {
        model_type: modelType,
        test_size: 0.2,
        random_state: 42
      };
      
      // 将重要参数添加到表单中
      const formParams = {};
      if (modelType === 'xgboost') {
        formParams.n_estimators = data.params.n_estimators || 100;
        formParams.learning_rate = data.params.learning_rate || 0.1;
        formParams.max_depth = data.params.max_depth || 5;
      } else if (['ridge', 'lasso', 'elasticnet'].includes(modelType)) {
        formParams.alpha = data.params.alpha || 1.0;
        if (modelType === 'elasticnet') {
          formParams.l1_ratio = data.params.l1_ratio || 0.5;
        }
      }
      
      modelForm.setFieldsValue({...initialValues, ...formParams});
    } catch (err) {
      notification.error({
        message: '获取模型参数失败',
        description: err.message
      });
    }
  };

  // 模型类型变更处理
  const handleModelTypeChange = (value) => {
    setSelectedModelType(value);
    fetchModelParams(value);
  };

  // 训练模型
  const handleTrainModel = async (values) => {
    try {
      setTrainLoading(true);
      
      // 提取模型参数
      const { model_type, test_size, random_state, ...params } = values;
      
      const requestBody = {
        model_type,
        test_size,
        random_state,
        params
      };
      
      const response = await fetch('/api/models/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '训练模型失败');
      }
      
      const data = await response.json();
      
      notification.success({
        message: '模型训练成功',
        description: data.message
      });
      
      // 刷新模型列表
      fetchModels();
      
      // 关闭对话框
      setTrainModalVisible(false);
      setTrainLoading(false);
    } catch (err) {
      notification.error({
        message: '训练模型失败',
        description: err.message
      });
      setTrainLoading(false);
    }
  };

  // 删除模型
  const handleDeleteModel = async (model) => {
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(model.path.replace('../model/', ''))}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '删除模型失败');
      }
      
      notification.success({
        message: '删除成功',
        description: `模型 ${model.name} 已成功删除`
      });
      
      // 刷新模型列表
      fetchModels();
    } catch (err) {
      notification.error({
        message: '删除模型失败',
        description: err.message
      });
    }
  };

  // 激活模型
  const handleActivateModel = async (model) => {
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(model.path.replace('../model/', ''))}` + '/activate', {
        method: 'POST'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '激活模型失败');
      }
      
      notification.success({
        message: '激活成功',
        description: `模型 ${model.name} 已成功激活`
      });
      
      // 设置当前激活的模型
      setActiveModel(model.path);
    } catch (err) {
      notification.error({
        message: '激活模型失败',
        description: err.message
      });
    }
  };

  // 查看模型详情
  const showModelDetail = (model) => {
    setSelectedModel(model);
    setModelDetailVisible(true);
  };

  // 表格列定义
  const columns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          <a onClick={() => showModelDetail(record)}>{text}</a>
          {activeModel === record.path && <Tag color="green">当前激活</Tag>}
        </Space>
      )
    },
    {
      title: '模型类型',
      dataIndex: 'type',
      key: 'type',
      render: text => {
        const typeColors = {
          'XGBoost': 'green',
          'Linear-linear': 'blue',
          'Linear-ridge': 'purple',
          'Linear-lasso': 'magenta',
          'Linear-elasticnet': 'orange'
        };
        return <Tag color={typeColors[text] || 'default'}>{text}</Tag>;
      }
    },
    {
      title: 'RMSE',
      dataIndex: ['metrics', 'rmse'],
      key: 'rmse',
      render: text => text?.toFixed(2) || '-'
    },
    {
      title: 'R²',
      dataIndex: ['metrics', 'r2'],
      key: 'r2',
      render: text => text?.toFixed(4) || '-'
    },
    {
      title: '创建时间',
      key: 'created_at',
      render: (_, record) => {
        try {
          const timestamp = record.path ? new Date(fs.statSync(record.path).mtime) : null;
          return timestamp ? moment(timestamp).format('YYYY-MM-DD HH:mm:ss') : '-';
        } catch {
          return '-';
        }
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="small">
          <Button 
            type="primary" 
            size="small" 
            icon={<CheckCircleOutlined />}
            onClick={() => handleActivateModel(record)}
            disabled={activeModel === record.path}
          >
            激活
          </Button>
          <Popconfirm
            title="确定要删除此模型吗？"
            okText="是"
            cancelText="否"
            onConfirm={() => handleDeleteModel(record)}
          >
            <Button type="danger" size="small" icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ];

  // 渲染函数
  return (
    <Layout className="layout">
      <Head>
        <title>模型管理 - PropertyWize</title>
      </Head>

      <MainHeader selectedKey="3" />

      <Content style={{ padding: '0 50px', marginTop: 64 }}>
        <div style={{ background: '#fff', padding: 24, minHeight: 'calc(100vh - 64px - 69px)', borderRadius: '4px', marginTop: '20px' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2}>模型管理</Title>
            </Col>
            <Col>
              <Button 
                type="primary" 
                icon={<PlusOutlined />} 
                onClick={() => {
                  setSelectedModelType('xgboost');
                  fetchModelParams('xgboost');
                  setTrainModalVisible(true);
                }}
              >
                训练新模型
              </Button>
            </Col>
          </Row>
          
          <Paragraph>
            管理和训练不同类型的房价预测模型，比较它们的性能，并选择最佳模型用于预测。
          </Paragraph>

          {loading ? (
            <div style={{ textAlign: 'center', margin: '50px 0' }}>
              <Spin size="large" />
              <p style={{ marginTop: 20 }}>加载模型数据...</p>
            </div>
          ) : error ? (
            <Alert
              message="加载错误"
              description={error}
              type="error"
              showIcon
            />
          ) : (
            <Table 
              columns={columns} 
              dataSource={models} 
              rowKey="path"
              pagination={{ pageSize: 10 }}
            />
          )}
        </div>
      </Content>

      <Footer style={{ textAlign: 'center' }}>
        PropertyWize ©{new Date().getFullYear()} 可解释房产估价系统
      </Footer>

      {/* 训练模型对话框 */}
      <Modal
        title="训练新模型"
        visible={trainModalVisible}
        onCancel={() => setTrainModalVisible(false)}
        footer={null}
        width={700}
      >
        <Form
          form={modelForm}
          layout="vertical"
          onFinish={handleTrainModel}
          initialValues={{
            model_type: 'xgboost',
            test_size: 0.2,
            random_state: 42
          }}
        >
          <Form.Item
            name="model_type"
            label="模型类型"
            rules={[{ required: true, message: '请选择模型类型' }]}
          >
            <Select onChange={handleModelTypeChange}>
              <Option value="xgboost">XGBoost（梯度提升树）</Option>
              <Option value="linear">线性回归</Option>
              <Option value="ridge">岭回归</Option>
              <Option value="lasso">Lasso回归</Option>
              <Option value="elasticnet">ElasticNet回归</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="test_size"
            label="测试集比例"
            rules={[{ required: true, message: '请输入测试集比例' }]}
          >
            <InputNumber min={0.1} max={0.5} step={0.05} style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item
            name="random_state"
            label="随机种子"
            rules={[{ required: true, message: '请输入随机种子' }]}
          >
            <InputNumber min={1} style={{ width: '100%' }} />
          </Form.Item>

          <Divider orientation="left">模型参数</Divider>

          {/* XGBoost参数 */}
          {selectedModelType === 'xgboost' && (
            <>
              <Form.Item
                name="n_estimators"
                label="树的数量"
                rules={[{ required: true, message: '请输入树的数量' }]}
              >
                <InputNumber min={10} max={1000} style={{ width: '100%' }} />
              </Form.Item>

              <Form.Item
                name="learning_rate"
                label="学习率"
                rules={[{ required: true, message: '请输入学习率' }]}
              >
                <InputNumber min={0.001} max={1} step={0.01} style={{ width: '100%' }} />
              </Form.Item>

              <Form.Item
                name="max_depth"
                label="树的最大深度"
                rules={[{ required: true, message: '请输入树的最大深度' }]}
              >
                <InputNumber min={1} max={15} style={{ width: '100%' }} />
              </Form.Item>
            </>
          )}

          {/* 线性模型参数 */}
          {['ridge', 'lasso', 'elasticnet'].includes(selectedModelType) && (
            <Form.Item
              name="alpha"
              label="正则化强度"
              rules={[{ required: true, message: '请输入正则化强度' }]}
            >
              <InputNumber min={0.0001} max={10} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
          )}

          {/* ElasticNet特有参数 */}
          {selectedModelType === 'elasticnet' && (
            <Form.Item
              name="l1_ratio"
              label="L1正则化比例"
              rules={[{ required: true, message: '请输入L1正则化比例' }]}
            >
              <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} />
            </Form.Item>
          )}

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={trainLoading} style={{ marginRight: 10 }}>
              开始训练
            </Button>
            <Button onClick={() => setTrainModalVisible(false)}>
              取消
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* 模型详情抽屉 */}
      <Drawer
        title="模型详情"
        placement="right"
        width={600}
        onClose={() => setModelDetailVisible(false)}
        visible={modelDetailVisible}
      >
        {selectedModel && (
          <>
            <Descriptions title="基本信息" bordered>
              <Descriptions.Item label="模型名称" span={3}>{selectedModel.name}</Descriptions.Item>
              <Descriptions.Item label="模型类型" span={3}>
                <Tag color="blue">{selectedModel.type}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态" span={3}>
                {activeModel === selectedModel.path ? 
                  <Badge status="success" text="当前激活" /> : 
                  <Badge status="default" text="未激活" />}
              </Descriptions.Item>
            </Descriptions>

            <Divider />

            <Descriptions title="性能指标" bordered>
              <Descriptions.Item label="RMSE" span={3}>
                {selectedModel.metrics?.rmse?.toFixed(4) || '-'}
              </Descriptions.Item>
              <Descriptions.Item label="MAE" span={3}>
                {selectedModel.metrics?.mae?.toFixed(4) || '-'}
              </Descriptions.Item>
              <Descriptions.Item label="R²" span={3}>
                {selectedModel.metrics?.r2?.toFixed(4) || '-'}
              </Descriptions.Item>
            </Descriptions>

            <Divider />

            <Title level={4}>模型参数</Title>
            <pre style={{ backgroundColor: '#f5f5f5', padding: 15, borderRadius: 4, overflow: 'auto' }}>
              {JSON.stringify(selectedModel.metadata?.params || {}, null, 2)}
            </pre>
            
            <Divider />
            
            <Space>
              <Button 
                type="primary" 
                icon={<CheckCircleOutlined />}
                onClick={() => handleActivateModel(selectedModel)}
                disabled={activeModel === selectedModel.path}
              >
                激活此模型
              </Button>
              <Popconfirm
                title="确定要删除此模型吗？"
                okText="是"
                cancelText="否"
                onConfirm={() => {
                  handleDeleteModel(selectedModel);
                  setModelDetailVisible(false);
                }}
              >
                <Button danger icon={<DeleteOutlined />}>
                  删除此模型
                </Button>
              </Popconfirm>
            </Space>
          </>
        )}
      </Drawer>
    </Layout>
  );
} 