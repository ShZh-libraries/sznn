const { join } = require("path");

module.exports = [{
    entry: './index.ts',
    module: {
        rules: [{
            test: /\*.ts$/,
            use: 'ts-loader',
        }],
    },
    resolve: {
        extensions: ['.ts', '.js'],
    },
    output: {
        filename: 'index.js',
        path: join(__dirname, 'out')
    }
}]