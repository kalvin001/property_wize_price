/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  transpilePackages: [
    '@ant-design/charts',
    '@antv/g2-extension-plot',
    '@antv/g2',
    '@antv/g',
    '@antv/util',
    '@antv/component',
    '@antv/hierarchy',
    'rc-pagination',
    'rc-util',
    'd3-hierarchy',
    'd3-selection',
    'd3-array',
    'd3-path',
    'd3-shape',
    'd3-interpolate',
    'd3-color',
    'antd',
    '@ant-design',
    'rc-picker'
  ],
  
  // 图像优化配置
  images: {
    domains: ['localhost'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    minimumCacheTTL: 60,
    formats: ['image/webp'],
  },
  
  // 添加API请求代理配置
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*' // 代理到后端8000端口
      }
    ]
  },
  
  // 添加webpack配置，允许导入ESM模块
  webpack: (config) => {
    // 禁用压缩，避免某些模块的压缩问题
    config.optimization.minimize = false;
    
    // 解决ESM和CommonJS混合使用的问题
    config.resolve.alias = {
      ...config.resolve.alias,
      'd3-interpolate': false,
      'd3-hierarchy': false,
      'd3-selection': false,
      'd3-array': false
    };
    
    // 配置节点处理
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
    };
    
    // 排除某些包的服务端渲染
    config.externals = [
      ...config.externals || [],
      {'@ant-design/charts': 'AntDesignCharts'}
    ];
    
    return config;
  },
  
  // 优化构建配置
  experimental: {
    optimizeCss: true, // CSS优化，现在已安装critters包
    scrollRestoration: true, // 启用滚动恢复
  }
};

module.exports = nextConfig; 