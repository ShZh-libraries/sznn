const path = require('path');
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = [
  {
    // For shorter compilation time
    mode: "development",
    devtool: "source-map",
    entry: path.join(__dirname, "index.ts"),
    module: {
      rules: [
        {
          test: /\.ts$/,
          use: "ts-loader",
        },
      ],
    },
    resolve: {
      extensions: [".ts", ".js"],
      fallback: {
        fs: false,
      },
    },
    plugins: [new NodePolyfillPlugin()],
    experiments: {
      asyncWebAssembly: true
    },
    output: {
      filename: "index.js",
      path: path.join(__dirname, "./dist"),
    },
    ignoreWarnings: [{
      message: /Circular dependency/
    }],
  },
];
