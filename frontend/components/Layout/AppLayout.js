import { Layout, Typography, Divider } from 'antd';
import MainHeader from '../../components/Header';
import Link from 'next/link';

const { Content, Footer } = Layout;
const { Title, Paragraph } = Typography;

const AppLayout = ({ children }) => {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <MainHeader />
      
      <Content style={{ padding: '0 50px', marginTop: 64 }}>
        {children}
      </Content>
      
      <Footer style={{ textAlign: 'center', background: '#001529', color: 'white', padding: '24px 50px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
          <div style={{ flex: '1', minWidth: '200px', marginBottom: '20px' }}>
            <Title level={4} style={{ color: 'white' }}>PropertyWize</Title>
            <Paragraph style={{ color: '#ccc' }}>
              AI驱动的智能房产估价系统，提供准确、透明的房产价值分析。
            </Paragraph>
          </div>
          
          <div style={{ flex: '1', minWidth: '200px', marginBottom: '20px' }}>
            <Title level={4} style={{ color: 'white' }}>快速链接</Title>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <Link href="/" style={{ color: '#ccc', marginBottom: '10px' }} legacyBehavior>首页</Link>
              <Link href="/model-center" style={{ color: '#ccc', marginBottom: '10px' }} legacyBehavior>模型中心</Link>
              <Link href="/property-reports" style={{ color: '#ccc', marginBottom: '10px' }} legacyBehavior>房产估价报告</Link>
              <a href="#" style={{ color: '#ccc' }}>联系我们</a>
            </div>
          </div>
          
          <div style={{ flex: '1', minWidth: '200px', marginBottom: '20px' }}>
            <Title level={4} style={{ color: 'white' }}>联系方式</Title>
            <Paragraph style={{ color: '#ccc' }}>
              邮箱: info@propertywize.com<br />
              电话: (021) 1234-5678<br />
              地址: 杭州市高新区
            </Paragraph>
          </div>
        </div>
        
        <Divider style={{ borderColor: 'rgba(255,255,255,0.2)' }} />
        <Paragraph style={{ color: '#ccc', marginBottom: 0 }}>
          PropertyWize © {new Date().getFullYear()} - 智能房产估价系统
        </Paragraph>
      </Footer>
    </Layout>
  );
};

export default AppLayout; 