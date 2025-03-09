import React, { useEffect } from 'react';

// 此组件仅在客户端导入Ant Design样式
const AntdStylesImport = () => {
  useEffect(() => {
    // 在客户端运行时动态导入
    import('antd/dist/reset.css');
  }, []);

  return null; // 此组件不渲染任何内容
};

export default AntdStylesImport; 