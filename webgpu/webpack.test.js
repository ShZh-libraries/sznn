const path = require('path');
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = {
    mode: "development",
    entry: {
        "test/abs": path.resolve(__dirname, "test", "abs.spec.ts"),
        "test/pad": path.resolve(__dirname, "test", "pad.spec.ts"),
        "test/conv": path.resolve(__dirname, "test", "conv.spec.ts"),
        "test/pool": path.resolve(__dirname, "test", "pooling.spec.ts"),
        "test/relu": path.resolve(__dirname, "test", "relu.spec.ts"),
    },
    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist'),
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                use: 'ts-loader',
                exclude: /node_modules/
            },
            {
                test: /\.wgsl$/,
                use: "ts-shader-loader",
            }
        ]
    },
    plugins: [
        new NodePolyfillPlugin()
    ],
    resolve: {
        extensions: [".js", '.ts'],
    }
};
