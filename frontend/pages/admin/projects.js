import { useState, useEffect } from 'react';
import { 
  Card, Table, Button, Space, Modal, Form, Input, message, Tabs, 
  Descriptions, Badge, Spin, Upload, Tooltip, Typography, List, Tag,
  Select, Switch, Progress, Empty
} from 'antd';
import { 
  PlusOutlined, EditOutlined, DeleteOutlined, ScanOutlined, 
  ExportOutlined, ImportOutlined, InfoCircleOutlined, UploadOutlined,
  CloudUploadOutlined, CloudServerOutlined, SyncOutlined, CodeOutlined
} from '@ant-design/icons';
import axios from 'axios';
import MainLayout from '../../components/Layout/MainLayout';

const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;
const { Option } = Select;
const { Dragger } = Upload;

const ProjectsPage = () => {
  const [loading, setLoading] = useState(false);
  const [deployLoading, setDeployLoading] = useState(false);
  const [deployProgress, setDeployProgress] = useState(0);
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [scanModalVisible, setScanModalVisible] = useState(false);
  const [importModalVisible, setImportModalVisible] = useState(false);
  const [deployModalVisible, setDeployModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [scanForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [deployForm] = Form.useForm();

  const fetchProjects = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/projects');
      setProjects(response.data);
    } catch (error) {
      console.error('获取项目列表失败:', error);
      message.error('获取项目列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const handleCreate = async (values) => {
    try {
      setLoading(true);
      await axios.post('/api/projects', values);
      message.success('项目创建成功');
      setCreateModalVisible(false);
      form.resetFields();
      fetchProjects();
    } catch (error) {
      console.error('创建项目失败:', error);
      message.error('创建项目失败');
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = async (values) => {
    try {
      setLoading(true);
      await axios.put(`/api/projects/${selectedProject.id}`, values);
      message.success('项目更新成功');
      setEditModalVisible(false);
      editForm.resetFields();
      fetchProjects();
    } catch (error) {
      console.error('更新项目失败:', error);
      message.error('更新项目失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个项目吗？此操作不可恢复。',
      okText: '确认',
      cancelText: '取消',
      onOk: async () => {
        try {
          setLoading(true);
          await axios.delete(`/api/projects/${id}`);
          message.success('项目删除成功');
          fetchProjects();
        } catch (error) {
          console.error('删除项目失败:', error);
          message.error('删除项目失败');
        } finally {
          setLoading(false);
        }
      },
    });
  };

  const handleScan = async (values) => {
    try {
      setLoading(true);
      await axios.post('/api/projects/scan', values);
      message.success('项目扫描成功');
      setScanModalVisible(false);
      scanForm.resetFields();
      fetchProjects();
    } catch (error) {
      console.error('扫描项目失败:', error);
      message.error('扫描项目失败');
    } finally {
      setLoading(false);
    }
  };

  const showEditModal = (project) => {
    setSelectedProject(project);
    editForm.setFieldsValue({
      name: project.name,
      description: project.description || '',
    });
    setEditModalVisible(true);
  };

  const showDetailModal = async (project) => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/projects/${project.id}`);
      setSelectedProject(response.data);
      setDetailModalVisible(true);
    } catch (error) {
      console.error('获取项目详情失败:', error);
      message.error('获取项目详情失败');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (id) => {
    try {
      setLoading(true);
      message.loading('正在导出项目...');
      window.location.href = `/api/projects/${id}/export`;
      message.success('项目导出成功');
    } catch (error) {
      console.error('导出项目失败:', error);
      message.error('导出项目失败');
    } finally {
      setLoading(false);
    }
  };

  const handleImport = async (info) => {
    if (info.file.status === 'uploading') {
      setLoading(true);
      return;
    }
    
    if (info.file.status === 'done') {
      message.success('项目导入成功');
      setImportModalVisible(false);
      fetchProjects();
      setLoading(false);
    } else if (info.file.status === 'error') {
      message.error('项目导入失败');
      setLoading(false);
    }
  };

  const showDeployModal = (project) => {
    setSelectedProject(project);
    deployForm.setFieldsValue({
      environment: 'development',
      auto_restart: true,
      build_frontend: true,
    });
    setDeployModalVisible(true);
  };

  const handleDeploy = async (values) => {
    try {
      setDeployLoading(true);
      setDeployProgress(0);
      
      // 显示部署进度
      const progressInterval = setInterval(() => {
        setDeployProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 1000);

      // 发起部署请求
      await axios.post(`/api/projects/${selectedProject.id}/deploy`, values);
      
      clearInterval(progressInterval);
      setDeployProgress(100);
      
      message.success('项目部署成功');
      setDeployModalVisible(false);
      deployForm.resetFields();
      
      // 更新项目状态
      fetchProjects();
    } catch (error) {
      console.error('部署项目失败:', error);
      message.error(`部署项目失败: ${error.response?.data?.detail || error.message}`);
    } finally {
      setDeployLoading(false);
    }
  };

  const handleUpdate = async (id) => {
    try {
      setLoading(true);
      await axios.post(`/api/projects/${id}/update`);
      message.success('项目更新成功');
      fetchProjects();
    } catch (error) {
      console.error('更新项目失败:', error);
      message.error(`更新项目失败: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '项目名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (text) => new Date(text).toLocaleString(),
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (text) => new Date(text).toLocaleString(),
    },
    {
      title: '操作',
      key: 'action',
      width: 400,
      render: (_, record) => (
        <Space size="small" wrap>
          <Button 
            type="primary" 
            icon={<InfoCircleOutlined />} 
            onClick={() => showDetailModal(record)}
          >
            详情
          </Button>
          <Button 
            icon={<EditOutlined />} 
            onClick={() => showEditModal(record)}
          >
            编辑
          </Button>
          <Button 
            icon={<SyncOutlined />} 
            onClick={() => handleUpdate(record.id)}
          >
            更新
          </Button>
          <Button 
            type="primary"
            icon={<CloudUploadOutlined />} 
            onClick={() => showDeployModal(record)}
          >
            部署
          </Button>
          <Button 
            icon={<ExportOutlined />} 
            onClick={() => handleExport(record.id)}
          >
            导出
          </Button>
          <Button 
            danger 
            icon={<DeleteOutlined />} 
            onClick={() => handleDelete(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  const renderProjectDetails = () => {
    if (!selectedProject) return null;

    const structure = selectedProject.config?.structure || {};
    const files = selectedProject.config?.files || [];

    return (
      <Tabs defaultActiveKey="1">
        <TabPane tab="基本信息" key="1">
          <Descriptions bordered column={2}>
            <Descriptions.Item label="项目ID">{selectedProject.id}</Descriptions.Item>
            <Descriptions.Item label="项目名称">{selectedProject.name}</Descriptions.Item>
            <Descriptions.Item label="创建时间">{new Date(selectedProject.created_at).toLocaleString()}</Descriptions.Item>
            <Descriptions.Item label="更新时间">{new Date(selectedProject.updated_at).toLocaleString()}</Descriptions.Item>
            <Descriptions.Item label="描述" span={2}>
              {selectedProject.description || '无描述'}
            </Descriptions.Item>
          </Descriptions>
        </TabPane>
        
        <TabPane tab="项目结构" key="2">
          <Card title="技术栈信息">
            <Descriptions bordered column={2}>
              <Descriptions.Item label="前端框架">
                {structure.frontend_framework}
                {structure.frontend_framework !== 'unknown' && <Badge status="success" />}
              </Descriptions.Item>
              <Descriptions.Item label="后端框架">
                {structure.backend_framework}
                {structure.backend_framework !== 'unknown' && <Badge status="success" />}
              </Descriptions.Item>
              <Descriptions.Item label="前端目录">
                {structure.frontend ? <Badge status="success" text="存在" /> : <Badge status="error" text="不存在" />}
              </Descriptions.Item>
              <Descriptions.Item label="后端目录">
                {structure.backend ? <Badge status="success" text="存在" /> : <Badge status="error" text="不存在" />}
              </Descriptions.Item>
              <Descriptions.Item label="数据库目录">
                {structure.database ? <Badge status="success" text="存在" /> : <Badge status="error" text="不存在" />}
              </Descriptions.Item>
              <Descriptions.Item label="模型目录">
                {structure.models ? <Badge status="success" text="存在" /> : <Badge status="error" text="不存在" />}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </TabPane>
        
        <TabPane tab="文件列表" key="3">
          <List
            loading={loading}
            itemLayout="horizontal"
            dataSource={files}
            pagination={{
              pageSize: 10,
              simple: true,
            }}
            renderItem={item => (
              <List.Item>
                <List.Item.Meta
                  title={
                    <Space>
                      <Text>{item.path}</Text>
                      <Tag color="blue">{item.type}</Tag>
                    </Space>
                  }
                  description={
                    <Space>
                      <Text type="secondary">大小: {(item.size / 1024).toFixed(2)} KB</Text>
                      <Text type="secondary">修改时间: {new Date(item.last_modified).toLocaleString()}</Text>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </TabPane>

        <TabPane tab="部署历史" key="4">
          {selectedProject.deployments && selectedProject.deployments.length > 0 ? (
            <List
              itemLayout="horizontal"
              dataSource={selectedProject.deployments || []}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    title={
                      <Space>
                        <Text>环境: {item.environment}</Text>
                        <Badge 
                          status={item.status === 'success' ? 'success' : 'error'} 
                          text={item.status === 'success' ? '成功' : '失败'} 
                        />
                      </Space>
                    }
                    description={
                      <Space direction="vertical">
                        <Text type="secondary">部署时间: {new Date(item.created_at).toLocaleString()}</Text>
                        {item.details && (
                          <Text type="secondary">详情: {item.details.message || JSON.stringify(item.details)}</Text>
                        )}
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          ) : (
            <Empty description="暂无部署记录" />
          )}
        </TabPane>
      </Tabs>
    );
  };

  return (
    <MainLayout selectedKey="admin-projects">
      <Title level={2}>项目管理</Title>
      <Paragraph>
        在这里您可以管理所有项目，包括创建、编辑、扫描、导出、导入、更新和部署项目。
      </Paragraph>
      
      <Card>
        <Space style={{ marginBottom: 16 }} wrap>
          <Button 
            type="primary" 
            icon={<PlusOutlined />} 
            onClick={() => setCreateModalVisible(true)}
          >
            创建项目
          </Button>
          <Button 
            icon={<ScanOutlined />} 
            onClick={() => setScanModalVisible(true)}
          >
            扫描项目
          </Button>
          <Button 
            icon={<ImportOutlined />} 
            onClick={() => setImportModalVisible(true)}
          >
            导入项目
          </Button>
        </Space>
        
        <Table 
          columns={columns} 
          dataSource={projects} 
          rowKey="id" 
          loading={loading}
          pagination={{ pageSize: 10 }}
          scroll={{ x: 'max-content' }}
        />
      </Card>
      
      {/* 创建项目的模态框 */}
      <Modal
        title="创建新项目"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreate}
        >
          <Form.Item
            name="name"
            label="项目名称"
            rules={[{ required: true, message: '请输入项目名称' }]}
          >
            <Input placeholder="请输入项目名称" />
          </Form.Item>
          <Form.Item
            name="description"
            label="项目描述"
          >
            <TextArea rows={4} placeholder="请输入项目描述" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                创建
              </Button>
              <Button onClick={() => setCreateModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
      
      {/* 编辑项目的模态框 */}
      <Modal
        title="编辑项目"
        open={editModalVisible}
        onCancel={() => setEditModalVisible(false)}
        footer={null}
      >
        <Form
          form={editForm}
          layout="vertical"
          onFinish={handleEdit}
        >
          <Form.Item
            name="name"
            label="项目名称"
            rules={[{ required: true, message: '请输入项目名称' }]}
          >
            <Input placeholder="请输入项目名称" />
          </Form.Item>
          <Form.Item
            name="description"
            label="项目描述"
          >
            <TextArea rows={4} placeholder="请输入项目描述" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                保存
              </Button>
              <Button onClick={() => setEditModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
      
      {/* 项目详情的模态框 */}
      <Modal
        title="项目详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width={800}
        footer={[
          <Button key="back" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>,
        ]}
      >
        {loading ? <Spin /> : renderProjectDetails()}
      </Modal>
      
      {/* 扫描项目的模态框 */}
      <Modal
        title="扫描项目"
        open={scanModalVisible}
        onCancel={() => setScanModalVisible(false)}
        footer={null}
      >
        <Form
          form={scanForm}
          layout="vertical"
          onFinish={handleScan}
        >
          <Paragraph>
            您可以选择通过项目ID或项目名称进行扫描。如果提供项目名称但不存在，将创建新项目。
          </Paragraph>
          <Form.Item
            name="project_id"
            label="项目ID"
          >
            <Input placeholder="请输入项目ID" />
          </Form.Item>
          <Form.Item
            name="name"
            label="项目名称"
          >
            <Input placeholder="请输入项目名称" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                开始扫描
              </Button>
              <Button onClick={() => setScanModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
      
      {/* 导入项目的模态框 */}
      <Modal
        title="导入项目"
        open={importModalVisible}
        onCancel={() => setImportModalVisible(false)}
        footer={[
          <Button key="back" onClick={() => setImportModalVisible(false)}>
            取消
          </Button>,
        ]}
      >
        <Paragraph>
          请选择项目ZIP文件进行导入。导入的项目将包含所有项目文件和配置信息。
        </Paragraph>
        <Dragger
          name="file"
          action="/api/projects/import"
          onChange={handleImport}
          multiple={false}
          accept=".zip"
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域进行上传</p>
          <p className="ant-upload-hint">只支持ZIP格式的项目文件</p>
        </Dragger>
      </Modal>

      {/* 部署项目的模态框 */}
      <Modal
        title="部署项目"
        open={deployModalVisible}
        onCancel={() => !deployLoading && setDeployModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={deployForm}
          layout="vertical"
          onFinish={handleDeploy}
        >
          <Paragraph>
            选择部署环境和相关配置，系统将自动完成项目的部署。
          </Paragraph>
          
          <Form.Item
            name="environment"
            label="部署环境"
            rules={[{ required: true, message: '请选择部署环境' }]}
          >
            <Select placeholder="请选择部署环境">
              <Option value="development">开发环境</Option>
              <Option value="production">生产环境</Option>
              <Option value="testing">测试环境</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="build_frontend"
            label="构建前端"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          
          <Form.Item
            name="auto_restart"
            label="自动重启服务"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          
          {deployLoading && (
            <Form.Item label="部署进度">
              <Progress percent={deployProgress} status={deployProgress < 100 ? "active" : "success"} />
            </Form.Item>
          )}
          
          <Form.Item>
            <Space>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={deployLoading}
                disabled={deployLoading}
                icon={<CloudServerOutlined />}
              >
                开始部署
              </Button>
              <Button 
                onClick={() => setDeployModalVisible(false)}
                disabled={deployLoading}
              >
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </MainLayout>
  );
};

export default ProjectsPage; 