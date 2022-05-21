module.exports = function (config) {
  config.set({
    frameworks: ["mocha", "chai", "karma-typescript"],
    files: [
      "src/**/*.ts",
      "../../packages/common/**/*.ts",
      "test/**/*.spec.ts",
    ],
    reporters: ["progress"],
    preprocessors: {
      "**/*.ts": "karma-typescript",
      "../../packages/common/**/*.ts": "karma-typescript",
    },
    port: 9876, // karma web server port
    colors: true,
    logLevel: config.LOG_INFO,
    browsers: ["GPUTestEnv"],
    customLaunchers: {
      GPUTestEnv: {
        base: "ChromeCanary",
        flags: ["--enable-unsafe-webgpu"],
      },
    },
    autoWatch: false,
    singleRun: true,
    concurrency: Infinity,
    karmaTypescriptConfig: {
      compilerOptions: {
        target: "es2016",
        moduleResolution: "node",
        sourceMap: true,
        outDir: "dist",
        esModuleInterop: true,
        forceConsistentCasingInFileNames: true,
        declaration: true,
        strict: true,
        skipLibCheck: true,
        typeRoots: ["./node_modules/@webgpu/types", "./node_modules/@types"],
      },
    },
  });
};
