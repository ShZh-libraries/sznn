const path = require('path');
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

module.exports = {
    mode: "development",
    devtool: "source-map",
    ignoreWarnings: [{
        message: /Circular dependency/
    }],
    entry: path.resolve(__dirname, "src", "bootstrap.ts"),
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
    // output: {
    //     path: path.resolve(__dirname, 'dist'),
    //     filename: 'index.js',
    // },
    plugins: [
        new WasmPackPlugin({
            crateDirectory: path.resolve(__dirname, "src", "rs"),
            extraArgs: "--target web",
            forceMode: "production",
        })
    ],
    experiments: {
        asyncWebAssembly: true
    }
};
