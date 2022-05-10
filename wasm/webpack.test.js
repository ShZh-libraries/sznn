const fs = require("fs");
const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

module.exports = {
    mode: "development",
    devtool: "inline-source-map",
    entry: () => {
        return new Promise(resolve => {
            let enties = {};
            const files = fs.readdirSync(path.resolve(__dirname, "test"));
            for (const filename of files) {
                if (filename.endsWith(".spec.ts")) {
                    const basename = path.basename(filename, ".ts");
                    const outname = "test/" + basename;

                    enties[outname] = path.resolve(__dirname, "test", filename);
                }
            }

            resolve(enties);
        });
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
            }
        ]
    },
    plugins: [
        new NodePolyfillPlugin(),
        new HtmlWebpackPlugin({
            title: "WASM backend test",
            template: "./test/public/index.html",
            filename: "./test/index.html",
            inject: "head",
            scriptLoading: "blocking"
        })
    ],
    resolve: {
        extensions: [".js", '.ts'],
    },
    experiments: {
        asyncWebAssembly: true
    }
};
