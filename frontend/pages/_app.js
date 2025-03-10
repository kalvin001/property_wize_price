import '../styles/globals.css';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/lib/locale/zh_CN';
import Head from 'next/head';

// 动态导入Ant Design样式以避免服务端渲染问题
import dynamic from 'next/dynamic';

// 仅在客户端导入antd样式
const AntdStylesImport = dynamic(() => import('../components/AntdStylesImport'), { ssr: false });

function MyApp({ Component, pageProps }) {
  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
        <meta name="description" content="PropertyWize - 可解释房产估价系统" />
        <meta name="theme-color" content="#1890ff" />
      </Head>
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