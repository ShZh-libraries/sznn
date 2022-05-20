const path = require('path');

module.exports = [
  {
    mode: "development",
    devtool: "source-map",
    entry: path.join(__dirname, "src", "index.ts"),
    module: {
      rules: [
        {
          test: /\.ts$/,
          use: "ts-loader",
        },
        {
          test: /\.wgsl$/,
          use: "ts-shader-loader",
        }
      ],
    },
    resolve: {
      extensions: [".ts", ".js"],
    },
    output: {
      filename: "index.js",
      path: path.join(__dirname, "dist"),
      library: "sznn_webgpu",
      libraryTarget: "umd"
    },
  },
];
