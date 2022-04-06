const { join } = require("path");

module.exports = [
  {
    mode: "development",
    devtool: "source-map",
    entry: "./src/index.ts",
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
    },
    output: {
      filename: "index.js",
      path: join(__dirname, "dist"),
      library: "sznn_js",
      libraryTarget: "umd"
    },
  },
];
