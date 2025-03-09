import '../styles/globals.css';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/lib/locale/zh_CN';

// 动态导入Ant Design样式以避免服务端渲染问题
import dynamic from 'next/dynamic';

// 仅在客户端导入antd样式
const AntdStylesImport = dynamic(() => import('../components/AntdStylesImport'), { ssr: false });

function MyApp({ Component, pageProps }) {
  return (
    <>
      <AntdStylesImport />
      <ConfigProvider locale={zhCN} theme={{
        token: {
          colorPrimary: '#1890ff',
        },
      }}>
        <Component {...pageProps} />
      </ConfigProvider>
    </>
  );
}

export default MyApp; 