const { join } = require("path");

module.exports = [
  {
    mode: "development",
    devtool: "source-map",
    entry: "./js/index.ts",
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
      // filename: "sznn.js",
      path: join(__dirname, "dist"),
      library: "sznn",
    },
  },
];
