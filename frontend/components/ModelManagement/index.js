import { useState, useEffect } from 'react';
import { 
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
  Space,
  Drawer,
  Descriptions,
  Badge,
  Popconfirm,
  Statistic
} from 'antd';
import { 
  PlusOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  SettingOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import moment from 'moment';

const { Title, Paragraph, Text } = Typography;
const { Option } = Select;

const ModelManagement = () => {
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
  const [modelTypes] = useState(['xgboost', 'linear', 'ridge', 'lasso', 'elasticnet', 'knn']);
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
      // 设置表单默认值
      const initialValues = {
        model_type: modelType,
        test_size: 0.2,
        random_state: 42
      };
      
      // 首先设置默认参数，以防API调用失败
      const defaultParams = {
        xgboost: {
          n_estimators: 100,
          learning_rate: 0.1,
          max_depth: 5
        },
        linear: {},
        ridge: {
          alpha: 1.0
        },
        lasso: {
          alpha: 1.0
        },
        elasticnet: {
          alpha: 1.0,
          l1_ratio: 0.5
        },
        knn: {
          n_neighbors: 5,
          weights: 'uniform',
          p: 2,
          leaf_size: 30
        }
      };
      
      let formParams = defaultParams[modelType] || {};
      
      try {
        const response = await fetch(`/api/models/${modelType}/params`);
        if (response.ok) {
          const data = await response.json();
          setModelParams(data.params || {});
          
          // 使用API返回的参数覆盖默认参数
          if (modelType === 'xgboost') {
            formParams.n_estimators = data.params.n_estimators || formParams.n_estimators;
            formParams.learning_rate = data.params.learning_rate || formParams.learning_rate;
            formParams.max_depth = data.params.max_depth || formParams.max_depth;
          } else if (['ridge', 'lasso', 'elasticnet'].includes(modelType)) {
            formParams.alpha = data.params.alpha || formParams.alpha;
            if (modelType === 'elasticnet') {
              formParams.l1_ratio = data.params.l1_ratio || formParams.l1_ratio;
            }
          } else if (modelType === 'knn') {
            formParams.n_neighbors = data.params.n_neighbors || formParams.n_neighbors;
            formParams.weights = data.params.weights || formParams.weights;
            formParams.p = data.params.p || formParams.p;
            formParams.leaf_size = data.params.leaf_size || formParams.leaf_size;
          }
        } else {
          // API返回错误，使用默认参数
          console.warn(`获取${modelType}参数失败，使用默认参数`);
          notification.warning({
            message: '获取模型参数失败',
            description: `无法从服务器获取${modelType}的参数信息，将使用默认参数。您仍然可以训练模型。`,
          });
        }
      } catch (apiErr) {
        // API调用异常，使用默认参数
        console.error('API调用失败:', apiErr);
        notification.warning({
          message: '获取模型参数失败',
          description: `无法连接到服务器获取参数信息，将使用默认参数。错误详情: ${apiErr.message}`,
        });
      }
      
      // 无论API是否成功，都设置表单值
      modelForm.setFieldsValue({...initialValues, ...formParams});
    } catch (err) {
      console.error('设置表单参数失败:', err);
      notification.error({
        message: '设置模型参数失败',
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
      
      // 确保所有数值参数是有效的JSON值（不是NaN或Infinity等）
      const sanitizeValue = (value) => {
        // 如果是数字
        if (typeof value === 'number') {
          // 检查是否为有效的JSON可序列化数字
          if (!Number.isFinite(value)) {
            console.warn(`检测到无效数值: ${value}，已替换为默认值`);
            return 0; // 替换为默认值
          }
        }
        // 如果是对象，递归处理所有属性
        else if (typeof value === 'object' && value !== null) {
          const result = {};
          for (const key in value) {
            result[key] = sanitizeValue(value[key]);
          }
          return result;
        }
        return value;
      };
      
      // 处理所有参数
      const sanitizedParams = sanitizeValue(params);
      
      const requestBody = {
        model_type,
        test_size: Number(test_size),
        random_state: Number(random_state),
        params: sanitizedParams
      };
      
      // 记录请求数据
      console.log('发送训练请求:', JSON.stringify(requestBody));
      
      // 显示训练开始通知
      const trainStartKey = 'trainStart';
      notification.info({
        key: trainStartKey,
        message: '模型训练开始',
        description: `开始训练${model_type}模型，请耐心等待...`,
        duration: 0
      });
      
      try {
        const response = await fetch('/api/models/train', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestBody)
        });
        
        // 关闭训练开始通知
        notification.close(trainStartKey);
        
        if (!response.ok) {
          let errorMessage = '训练模型失败';
          try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorMessage;
          } catch (jsonError) {
            console.error('解析错误响应失败:', jsonError);
            if (response.status === 500) {
              errorMessage = '服务器内部错误，模型训练失败。可能是参数无效或数据问题导致。';
            } else {
              errorMessage = `服务器响应错误(${response.status}): ${response.statusText}`;
            }
          }
          
          throw new Error(errorMessage);
        }
        
        const data = await response.json();
        
        notification.success({
          message: '模型训练成功',
          description: data.message || `${model_type}模型训练完成，现在可以在模型列表中查看。`
        });
        
        // 刷新模型列表
        fetchModels();
        
        // 关闭对话框
        setTrainModalVisible(false);
      } catch (apiError) {
        console.error('API调用失败:', apiError);
        notification.error({
          message: '训练模型失败',
          description: `模型训练过程中出现错误: ${apiError.message}`,
          duration: 8
        });
      }
      
      setTrainLoading(false);
    } catch (err) {
      console.error('训练模型过程中发生异常:', err);
      notification.error({
        message: '训练模型失败',
        description: err.message,
        duration: 8
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
      sorter: (a, b) => a.name.localeCompare(b.name)
    },
    {
      title: '模型类型',
      dataIndex: 'type',
      key: 'type',
      filters: [
        { text: 'XGBoost', value: 'xgboost' },
        { text: '线性回归', value: 'linear' },
        { text: 'Ridge', value: 'ridge' },
        { text: 'Lasso', value: 'lasso' },
        { text: 'ElasticNet', value: 'elasticnet' },
        { text: 'KNN', value: 'knn' }
      ],
      onFilter: (value, record) => record.type === value,
      render: (type) => {
        const typeMap = {
          'xgboost': { color: 'green', name: 'XGBoost' },
          'linear': { color: 'blue', name: '线性回归' },
          'ridge': { color: 'purple', name: 'Ridge' },
          'lasso': { color: 'orange', name: 'Lasso' },
          'elasticnet': { color: 'cyan', name: 'ElasticNet' },
          'knn': { color: 'magenta', name: 'KNN' }
        };
        
        const modelType = typeMap[type] || { color: 'default', name: type };
        
        return <Tag color={modelType.color}>{modelType.name}</Tag>;
      }
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      sorter: (a, b) => new Date(a.created_at) - new Date(b.created_at),
      render: (text) => moment(text).format('YYYY-MM-DD HH:mm:ss')
    },
    {
      title: 'R²得分',
      dataIndex: ['metrics', 'r2_score'],
      key: 'r2_score',
      sorter: (a, b) => (a.metrics?.r2_score || 0) - (b.metrics?.r2_score || 0),
      render: (text) => text ? text.toFixed(4) : '-'
    },
    {
      title: 'RMSE',
      dataIndex: ['metrics', 'rmse'],
      key: 'rmse',
      sorter: (a, b) => (a.metrics?.rmse || 0) - (b.metrics?.rmse || 0),
      render: (text) => text ? text.toFixed(2) : '-'
    },
    {
      title: '状态',
      key: 'status',
      render: (_, record) => (
        <span>
          {activeModel === record.path ? (
            <Badge status="processing" text="当前激活" />
          ) : (
            <Badge status="default" text="未激活" />
          )}
        </span>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            onClick={() => showModelDetail(record)}
            icon={<InfoCircleOutlined />}
          >
            详情
          </Button>
          {activeModel !== record.path && (
            <Button
              type="link"
              size="small"
              onClick={() => handleActivateModel(record)}
              icon={<CheckCircleOutlined />}
            >
              激活
            </Button>
          )}
          <Popconfirm
            title="确定要删除此模型吗？"
            description="删除后无法恢复！"
            okText="确定"
            cancelText="取消"
            onConfirm={() => handleDeleteModel(record)}
          >
            <Button
              type="link"
              danger
              size="small"
              icon={<DeleteOutlined />}
              disabled={activeModel === record.path}
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Title level={2}>模型管理</Title>
        <Paragraph>
          这里您可以训练新的房价预测模型，管理现有模型，以及设置当前激活使用的模型。
        </Paragraph>
      </div>

      <Row justify="space-between" style={{ marginBottom: 16 }}>
        <Col>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => {
              setTrainModalVisible(true);
              fetchModelParams(selectedModelType);
            }}
          >
            训练新模型
          </Button>
        </Col>
      </Row>

      {loading ? (
        <div style={{ textAlign: 'center', margin: '50px 0' }}>
          <Spin size="large" />
          <p style={{ marginTop: 20 }}>加载模型列表...</p>
        </div>
      ) : error ? (
        <Alert
          message="加载错误"
          description={error}
          type="error"
          showIcon
        />
      ) : (
        <Card>
          <Table 
            columns={columns} 
            dataSource={models}
            rowKey="path"
            pagination={{ pageSize: 10 }}
          />
        </Card>
      )}

      {/* 训练模型对话框 */}
      <Modal
        title="训练新模型"
        open={trainModalVisible}
        onCancel={() => setTrainModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={modelForm}
          name="modelTrainingForm"
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
              <Option value="ridge">Ridge回归（L2正则化）</Option>
              <Option value="lasso">Lasso回归（L1正则化）</Option>
              <Option value="elasticnet">ElasticNet回归（L1+L2正则化）</Option>
              <Option value="knn">KNN（K最近邻）</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="test_size"
            label="测试集比例"
            rules={[{ required: true, message: '请输入测试集比例' }]}
          >
            <InputNumber
              min={0.1}
              max={0.5}
              step={0.05}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item
            name="random_state"
            label="随机种子"
            rules={[{ required: true, message: '请输入随机种子' }]}
          >
            <InputNumber
              min={0}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Divider orientation="left">模型参数</Divider>

          {selectedModelType === 'xgboost' && (
            <>
              <Form.Item
                name="n_estimators"
                label="树的数量(n_estimators)"
                tooltip="集成中决策树的数量，增加可提高模型性能但也增加训练时间"
                rules={[{ required: true, message: '请输入树的数量' }]}
              >
                <InputNumber
                  min={10}
                  max={1000}
                  step={10}
                  style={{ width: '100%' }}
                />
              </Form.Item>

              <Form.Item
                name="learning_rate"
                label="学习率(learning_rate)"
                tooltip="每棵树对最终预测的贡献比例，较小的值需要更多树但通常表现更好"
                rules={[{ required: true, message: '请输入学习率' }]}
              >
                <InputNumber
                  min={0.01}
                  max={1}
                  step={0.01}
                  style={{ width: '100%' }}
                />
              </Form.Item>

              <Form.Item
                name="max_depth"
                label="最大深度(max_depth)"
                tooltip="单棵树的最大深度，较大的值可能导致过拟合"
                rules={[{ required: true, message: '请输入最大深度' }]}
              >
                <InputNumber
                  min={1}
                  max={15}
                  style={{ width: '100%' }}
                />
              </Form.Item>
              
              <Alert
                message="XGBoost参数说明"
                description={
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    <li><strong>树的数量</strong>: 更多的树通常能提高模型性能，但会增加训练和预测时间</li>
                    <li><strong>学习率</strong>: 值越小，模型学习越慢但可能更精确，通常与树的数量成反比调整</li>
                    <li><strong>最大深度</strong>: 控制单棵树的复杂度，过大的值会导致过拟合，通常在3-10之间效果较好</li>
                  </ul>
                }
                type="info"
                showIcon
              />
            </>
          )}

          {['ridge', 'lasso', 'elasticnet'].includes(selectedModelType) && (
            <>
              <Form.Item
                name="alpha"
                label="正则化强度(alpha)"
                tooltip="控制模型复杂度的惩罚系数，较大的值会使模型更简单"
                rules={[{ required: true, message: '请输入正则化强度' }]}
              >
                <InputNumber
                  min={0.001}
                  max={10}
                  step={0.001}
                  style={{ width: '100%' }}
                />
              </Form.Item>
              
              {selectedModelType === 'elasticnet' && (
                <Form.Item
                  name="l1_ratio"
                  label="L1比例(l1_ratio)"
                  tooltip="L1正则化在总正则化中的比例，0表示纯L2正则化，1表示纯L1正则化"
                  rules={[{ required: true, message: '请输入L1比例' }]}
                >
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              )}
              
              <Alert
                message={`${selectedModelType === 'ridge' ? 'Ridge' : selectedModelType === 'lasso' ? 'Lasso' : 'ElasticNet'}参数说明`}
                description={
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    <li><strong>正则化强度(alpha)</strong>: 控制模型的复杂度，值越大模型越简单，特征系数越小</li>
                    {selectedModelType === 'ridge' && <li>Ridge使用L2正则化，倾向于使所有特征系数变小但不为零</li>}
                    {selectedModelType === 'lasso' && <li>Lasso使用L1正则化，倾向于产生稀疏解，使部分特征系数变为零</li>}
                    {selectedModelType === 'elasticnet' && (
                      <>
                        <li><strong>L1比例(l1_ratio)</strong>: 控制L1和L2正则化的混合比例</li>
                        <li>接近0时类似Ridge，接近1时类似Lasso，结合了两者的优点</li>
                      </>
                    )}
                  </ul>
                }
                type="info"
                showIcon
              />
            </>
          )}

          {selectedModelType === 'knn' && (
            <>
              <Form.Item
                name="n_neighbors"
                label="邻居数量(n_neighbors)"
                tooltip="预测时考虑的最近邻点数量，较大的值会使决策边界更平滑"
                rules={[{ required: true, message: '请输入邻居数量' }]}
              >
                <InputNumber
                  min={1}
                  max={50}
                  step={1}
                  style={{ width: '100%' }}
                />
              </Form.Item>

              <Form.Item
                name="weights"
                label="权重类型(weights)"
                tooltip="uniform表示所有点权重相等，distance表示权重与距离成反比"
                rules={[{ required: true, message: '请选择权重类型' }]}
              >
                <Select>
                  <Option value="uniform">Uniform（等权重）</Option>
                  <Option value="distance">Distance（距离加权）</Option>
                </Select>
              </Form.Item>

              <Form.Item
                name="p"
                label="距离度量(p)"
                tooltip="用于Minkowski距离计算的p参数，p=1为曼哈顿距离，p=2为欧氏距离"
                rules={[{ required: true, message: '请选择距离度量' }]}
              >
                <Select>
                  <Option value={1}>Manhattan距离 (p=1)</Option>
                  <Option value={2}>欧氏距离 (p=2)</Option>
                  <Option value={3}>Minkowski距离 (p=3)</Option>
                </Select>
              </Form.Item>
              
              <Form.Item
                name="leaf_size"
                label="叶子节点大小(leaf_size)"
                tooltip="数值越大，构建和查询KD树越快，但准确性可能下降"
                rules={[{ required: true, message: '请输入叶子节点大小' }]}
              >
                <InputNumber
                  min={1}
                  max={100}
                  step={1}
                  style={{ width: '100%' }}
                />
              </Form.Item>

              <Alert
                message="KNN参数说明"
                description={
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    <li><strong>邻居数量</strong>: 较小的值使模型对局部变化更敏感，较大的值使预测更稳定但可能丢失局部模式</li>
                    <li><strong>权重类型</strong>: 距离加权在处理噪声数据时通常表现更好</li>
                    <li><strong>距离度量</strong>: 欧氏距离(p=2)适用于连续特征，曼哈顿距离(p=1)对异常值更不敏感</li>
                    <li><strong>叶子节点大小</strong>: 影响算法速度，但不会显著影响预测性能</li>
                  </ul>
                }
                type="info"
                showIcon
              />
            </>
          )}

          <Form.Item>
            <Row justify="end" gutter={8}>
              <Col>
                <Button onClick={() => setTrainModalVisible(false)}>
                  取消
                </Button>
              </Col>
              <Col>
                <Button type="primary" htmlType="submit" loading={trainLoading}>
                  开始训练
                </Button>
              </Col>
            </Row>
          </Form.Item>
        </Form>
      </Modal>

      {/* 模型详情抽屉 */}
      <Drawer
        title={`模型详情 - ${selectedModel?.name || ''}`}
        placement="right"
        closable={true}
        onClose={() => setModelDetailVisible(false)}
        open={modelDetailVisible}
        width={600}
      >
        {selectedModel && (
          <>
            <Descriptions bordered column={1}>
              <Descriptions.Item label="模型名称">
                {selectedModel.name}
              </Descriptions.Item>
              <Descriptions.Item label="模型类型">
                {selectedModel.type === 'xgboost' && 'XGBoost（梯度提升树）'}
                {selectedModel.type === 'linear' && '线性回归'}
                {selectedModel.type === 'ridge' && 'Ridge回归（L2正则化）'}
                {selectedModel.type === 'lasso' && 'Lasso回归（L1正则化）'}
                {selectedModel.type === 'elasticnet' && 'ElasticNet回归（L1+L2正则化）'}
                {selectedModel.type === 'knn' && 'KNN（K最近邻）'}
              </Descriptions.Item>
              <Descriptions.Item label="模型路径">
                {selectedModel.path}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {moment(selectedModel.created_at).format('YYYY-MM-DD HH:mm:ss')}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {activeModel === selectedModel.path ? (
                  <Badge status="processing" text="当前激活" />
                ) : (
                  <Badge status="default" text="未激活" />
                )}
              </Descriptions.Item>
            </Descriptions>

            <Divider orientation="left">性能指标</Divider>

            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card>
                  <Statistic
                    title="R²得分"
                    value={selectedModel.metrics?.r2_score?.toFixed(4) || '-'}
                    precision={4}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card>
                  <Statistic
                    title="RMSE"
                    value={selectedModel.metrics?.rmse?.toFixed(2) || '-'}
                    precision={2}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card>
                  <Statistic
                    title="MAE"
                    value={selectedModel.metrics?.mae?.toFixed(2) || '-'}
                    precision={2}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card>
                  <Statistic
                    title="中位百分比误差"
                    value={selectedModel.metrics?.median_percentage_error?.toFixed(2) || '-'}
                    precision={2}
                    suffix="%"
                  />
                </Card>
              </Col>
            </Row>

            <Divider />

            <Row justify="space-between">
              <Col>
                <Button
                  type="primary"
                  disabled={activeModel === selectedModel.path}
                  onClick={() => {
                    handleActivateModel(selectedModel);
                    setModelDetailVisible(false);
                  }}
                >
                  激活此模型
                </Button>
              </Col>
              <Col>
                <Popconfirm
                  title="确定要删除此模型吗？"
                  description="删除后无法恢复！"
                  okText="确定"
                  cancelText="取消"
                  onConfirm={() => {
                    handleDeleteModel(selectedModel);
                    setModelDetailVisible(false);
                  }}
                  disabled={activeModel === selectedModel.path}
                >
                  <Button
                    danger
                    disabled={activeModel === selectedModel.path}
                  >
                    删除此模型
                  </Button>
                </Popconfirm>
              </Col>
            </Row>
          </>
        )}
      </Drawer>
    </div>
  );
};

export default ModelManagement; 