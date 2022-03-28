const { join } = require("path");
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = [{
    mode: 'development',
    devtool: 'source-map',
    entry: './examples/classify/index.ts',
    module: {
        rules: [{
            test: /\.ts$/,
            use: 'ts-loader',
        }],
    },
    resolve: {
        extensions: ['.ts', '.js'],
        fallback: {
            'fs': false
        }
    },
    plugins: [
        new NodePolyfillPlugin()
    ],
    output: {
        filename: 'index.js',
        path: join(__dirname, './dist')
    }
}];