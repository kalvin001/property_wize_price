import { useState } from 'react';
import Head from 'next/head';
import { 
  Layout, 
  Menu, 
  Typography, 
  Button, 
  Input, 
  Row, 
  Col, 
  Card, 
  Carousel, 
  Statistic, 
  Divider,
  Space
} from 'antd';
import { 
  SearchOutlined, 
  HomeOutlined, 
  BarChartOutlined, 
  FileTextOutlined,
  CheckCircleOutlined,
  RocketOutlined,
  BulbOutlined
} from '@ant-design/icons';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;
const { Search } = Input;

export default function Home() {
  const [searchValue, setSearchValue] = useState('');

  return (
    <Layout className="layout">
      <Head>
        <title>PropertyWize - 可解释房产估价系统</title>
        <meta name="description" content="基于AI的智能房产估价系统，提供详细的可解释性报告" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Header style={{ position: 'fixed', zIndex: 1, width: '100%', background: 'white', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
        <div style={{ float: 'left', marginRight: '30px' }}>
          <Typography.Title level={3} style={{ margin: '8px 0', color: '#1890ff' }}>
            <HomeOutlined /> PropertyWize
          </Typography.Title>
        </div>
        <Menu 
          theme="light" 
          mode="horizontal" 
          defaultSelectedKeys={['1']}
          style={{ lineHeight: '64px' }}
          items={[
            { key: '1', label: '首页' },
            { key: '2', label: '估价服务' },
            { key: '3', label: '样本报告' },
            { key: '4', label: '关于我们' },
          ]}
        />
      </Header>

      <Content style={{ padding: '0 50px', marginTop: 64 }}>
        {/* 1. 宣传横幅 */}
        <div className="hero-banner" style={{ 
          height: '500px', 
          background: 'linear-gradient(135deg, #1890ff 0%, #001529 100%)',
          display: 'flex',
          alignItems: 'center',
          borderRadius: '8px',
          marginTop: '30px',
          overflow: 'hidden',
          position: 'relative',
        }}>
          <div style={{ 
            position: 'absolute', 
            width: '100%', 
            height: '100%', 
            backgroundColor: 'rgba(0,0,0,0.3)',
            zIndex: 1
          }}></div>
          <div style={{ 
            position: 'absolute',
            width: '50%',
            left: '10%',
            zIndex: 2,
            color: 'white'
          }}>
            <Title style={{ color: 'white', fontSize: '48px', marginBottom: '20px' }}>
              智能房产估价
              <br />
              让决策更有把握
            </Title>
            <Paragraph style={{ color: 'white', fontSize: '18px', marginBottom: '30px' }}>
              基于人工智能和大数据分析，PropertyWize提供准确的房产估价与详细的解释报告，
              帮助您了解影响房价的每一个因素。
            </Paragraph>
            <Space size="large">
              <Button type="primary" size="large" style={{ height: '50px', fontSize: '16px', padding: '0 30px' }}>
                开始估价
              </Button>
              <Button ghost size="large" style={{ height: '50px', fontSize: '16px', padding: '0 30px', color: 'white', borderColor: 'white' }}>
                查看样本报告
              </Button>
            </Space>
          </div>
          <div style={{ 
            position: 'absolute',
            right: '-5%',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '55%',
            height: '80%',
            background: 'linear-gradient(45deg, #001529 0%, #1890ff 100%)',
            borderRadius: '8px',
            boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
            zIndex: 0
          }}></div>
        </div>

        {/* 2. 搜索区域 */}
        <div style={{ textAlign: 'center', margin: '60px 0' }}>
          <Title level={2}>查询您的房产估价报告</Title>
          <Paragraph style={{ fontSize: '16px', marginBottom: '30px' }}>
            输入报告ID或地址查询已生成的估价报告
          </Paragraph>
          <Row justify="center">
            <Col xs={24} sm={24} md={16} lg={12} xl={8}>
              <Search
                placeholder="输入报告ID或房产地址"
                enterButton="搜索报告"
                size="large"
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                onSearch={(value) => console.log(value)}
                style={{ width: '100%' }}
              />
            </Col>
          </Row>
        </div>

        {/* 3. 我们的优势 */}
        <div style={{ margin: '80px 0' }}>
          <Title level={2} style={{ textAlign: 'center', marginBottom: '50px' }}>为什么选择 PropertyWize</Title>
          
          <Row gutter={[32, 32]}>
            <Col xs={24} sm={12} md={8}>
              <Card 
                hoverable 
                style={{ height: '100%', textAlign: 'center', padding: '20px' }}
              >
                <BulbOutlined style={{ fontSize: '48px', color: '#1890ff', margin: '20px 0' }} />
                <Card.Meta
                  title="AI驱动的准确估价"
                  description="基于先进的机器学习算法和海量数据，提供准确的房产价值估算"
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={12} md={8}>
              <Card 
                hoverable 
                style={{ height: '100%', textAlign: 'center', padding: '20px' }}
              >
                <BarChartOutlined style={{ fontSize: '48px', color: '#1890ff', margin: '20px 0' }} />
                <Card.Meta
                  title="详细的可解释性分析"
                  description="透明展示影响房价的各项因素及其权重，让您了解每一分钱的来源"
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={12} md={8}>
              <Card 
                hoverable 
                style={{ height: '100%', textAlign: 'center', padding: '20px' }}
              >
                <FileTextOutlined style={{ fontSize: '48px', color: '#1890ff', margin: '20px 0' }} />
                <Card.Meta
                  title="专业的估价报告"
                  description="生成全面专业的PDF报告，包含详尽的房产分析和市场比较数据"
                />
              </Card>
            </Col>
          </Row>
        </div>

        {/* 4. 数据统计 */}
        <div style={{ 
          margin: '80px 0', 
          padding: '60px 0',
          background: '#f0f5ff',
          borderRadius: '8px',
          textAlign: 'center'
        }}>
          <Title level={2} style={{ marginBottom: '60px' }}>可信赖的房产估价服务</Title>
          
          <Row gutter={[48, 48]} justify="center">
            <Col xs={24} sm={12} md={6}>
              <Statistic title="估价准确率" value={96.8} suffix="%" />
              <Paragraph>高于行业平均水平</Paragraph>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic title="已处理房产" value={125000} />
              <Paragraph>来自全国各地的房产数据</Paragraph>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic title="考虑因素" value={50} suffix="+" />
              <Paragraph>综合分析影响房价的因素</Paragraph>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic title="客户满意度" value={98.2} suffix="%" />
              <Paragraph>基于10000+用户反馈</Paragraph>
            </Col>
          </Row>
        </div>

        {/* 5. 样本报告展示 */}
        <div style={{ margin: '80px 0' }}>
          <Title level={2} style={{ textAlign: 'center', marginBottom: '20px' }}>样本报告展示</Title>
          <Paragraph style={{ textAlign: 'center', fontSize: '16px', marginBottom: '40px' }}>
            查看我们的样本报告，了解PropertyWize如何分析房产价值
          </Paragraph>
          
          <Row gutter={[24, 24]}>
            <Col xs={24} sm={24} md={8}>
              <Card
                hoverable
                cover={<div style={{ 
                  height: '200px', 
                  background: '#f0f5ff',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}>
                  <FileTextOutlined style={{ fontSize: '64px', color: '#1890ff' }} />
                </div>}
              >
                <Card.Meta
                  title="公寓型房产估价报告"
                  description="市中心两室一厅公寓的详细估价分析"
                />
                <Button type="link" style={{ padding: 0, marginTop: '15px' }}>
                  查看样本 →
                </Button>
              </Card>
            </Col>
            
            <Col xs={24} sm={24} md={8}>
              <Card
                hoverable
                cover={<div style={{ 
                  height: '200px', 
                  background: '#f0f5ff',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}>
                  <FileTextOutlined style={{ fontSize: '64px', color: '#1890ff' }} />
                </div>}
              >
                <Card.Meta
                  title="别墅型房产估价报告"
                  description="郊区独栋别墅的详细估价分析"
                />
                <Button type="link" style={{ padding: 0, marginTop: '15px' }}>
                  查看样本 →
                </Button>
              </Card>
            </Col>
            
            <Col xs={24} sm={24} md={8}>
              <Card
                hoverable
                cover={<div style={{ 
                  height: '200px', 
                  background: '#f0f5ff',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}>
                  <FileTextOutlined style={{ fontSize: '64px', color: '#1890ff' }} />
                </div>}
              >
                <Card.Meta
                  title="商业地产估价报告"
                  description="商业地产投资价值分析报告"
                />
                <Button type="link" style={{ padding: 0, marginTop: '15px' }}>
                  查看样本 →
                </Button>
              </Card>
            </Col>
          </Row>
        </div>

        {/* 6. 行动召唤 */}
        <div style={{ 
          margin: '80px 0', 
          padding: '60px',
          background: 'linear-gradient(135deg, #f6ffed 0%, #e6f7ff 100%)',
          borderRadius: '8px',
          textAlign: 'center'
        }}>
          <Title level={2}>开始您的房产估价</Title>
          <Paragraph style={{ fontSize: '16px', maxWidth: '700px', margin: '20px auto 30px' }}>
            无论您是准备购买、出售房产，还是希望了解当前房产的市场价值，
            PropertyWize都能为您提供准确、透明的房产估价服务。
          </Paragraph>
          <Button type="primary" size="large" style={{ height: '50px', fontSize: '16px', padding: '0 40px' }}>
            立即开始
          </Button>
        </div>
      </Content>

      <Footer style={{ textAlign: 'center', background: '#001529', color: 'white', padding: '24px 50px' }}>
        <Row gutter={[48, 32]}>
          <Col xs={24} sm={24} md={8}>
            <Title level={4} style={{ color: 'white' }}>PropertyWize</Title>
            <Paragraph style={{ color: '#ccc' }}>
              AI驱动的智能房产估价系统，提供准确、透明的房产价值分析。
            </Paragraph>
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Title level={4} style={{ color: 'white' }}>快速链接</Title>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <a href="#" style={{ color: '#ccc', marginBottom: '10px' }}>首页</a>
              <a href="#" style={{ color: '#ccc', marginBottom: '10px' }}>估价服务</a>
              <a href="#" style={{ color: '#ccc', marginBottom: '10px' }}>样本报告</a>
              <a href="#" style={{ color: '#ccc' }}>联系我们</a>
            </div>
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Title level={4} style={{ color: 'white' }}>联系方式</Title>
            <Paragraph style={{ color: '#ccc' }}>
              邮箱: info@propertywize.com<br />
              电话: (021) 1234-5678<br />
              地址: 上海市浦东新区科技园区
            </Paragraph>
          </Col>
        </Row>
        <Divider style={{ borderColor: 'rgba(255,255,255,0.2)' }} />
        <Paragraph style={{ color: '#ccc', marginBottom: 0 }}>
          PropertyWize © {new Date().getFullYear()} - 智能房产估价系统
        </Paragraph>
      </Footer>
    </Layout>
  );
} 