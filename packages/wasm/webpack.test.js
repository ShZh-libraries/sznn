const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = {
  mode: "development",
  devtool: "inline-source-map",
  ignoreWarnings: [
    {
      message: /Circular dependency/,
    },
  ],
  entry: path.resolve(__dirname, "test", "index.ts"),
  output: {
    filename: "test.js",
    path: path.resolve(__dirname, "dist"),
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [
    new NodePolyfillPlugin(),
    new HtmlWebpackPlugin({
      title: "WASM backend test",
      template: "./test/public/index.html",
      filename: "./test/index.html",
      inject: "head",
      scriptLoading: "blocking",
    }),
  ],
  resolve: {
    extensions: [".js", ".ts"],
  },
  experiments: {
    asyncWebAssembly: true,
  },
};
