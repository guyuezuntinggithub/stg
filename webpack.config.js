// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const WebpackObfuscator = require('webpack-obfuscator');

module.exports = {
  // 模式: 'production' 会开启默认的优化
  mode: 'production',

  // 入口文件
  entry: './src/GSViewer.js',

  // 输出配置
  output: {
    path: path.resolve(__dirname, 'dist'),
    // 修改输出文件名，使其更像一个库
    filename: 'immerscape.min.js', 
    clean: true,
    
    // --- 新增 library 配置 ---
    library: {
      name: 'ImmerScapeViewer', // 库的名称，它将作为全局变量暴露
      type: 'umd',           // 库的类型，'umd' 通用性最强
      export: 'default',     // 指定将模块的 default 导出作为库的接口
    }
  },

  //externals: {
  //  three: {
  //    commonjs: 'three',
  //    commonjs2: 'three',
  //    amd: 'three',
  //    root: 'THREE', // 当通过 <script> 标签引入时，它会寻找全局变量 THREE
  //  },
  //},
  
  // 优化配置
  optimization: {
    minimize: true, // 开启最小化（压缩）
    minimizer: [
      new TerserPlugin(), // 使用 Terser 压缩 JS
    ],
  },

  // 插件列表
  plugins: [
    // 自动生成 HTML
    new HtmlWebpackPlugin({
      template: './index.html', // 使用现有的 HTML 文件作为模板
    }),

    new CopyPlugin({
      patterns: [
        // 将 'public' 文件夹的所有内容复制到 'dist' 文件夹的根目录
        { from: 'src/sorter/wasm', to: 'wasm' }, 
        { from: 'scenes', to: 'scenes' }, 
      ],
    }),
    
    // ⚠️ 混淆插件，只在生产模式下应用
    // WebpackObfuscator 必须放在所有插件的最后
    new WebpackObfuscator({
      rotateStringArray: true, // 旋转字符串数组
      stringArray: true,
      // 更多选项请参考 webpack-obfuscator 官方文档
    }, [
        // 你不希望被混淆的文件可以在这里排除
        // 'excluded_bundle_name.js' 
    ])
  ],

};