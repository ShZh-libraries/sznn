const path = require('path');
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = {
    mode: "development",
    entry: {
        "test/example": path.resolve(__dirname, "test", "abs.spec.ts"),
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
