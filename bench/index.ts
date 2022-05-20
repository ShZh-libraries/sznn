import { loadImage, displayImage } from "./utils/cv";
import { imagenetClasses } from "./utils/image_net";
import * as echarts from "echarts";
import { inferenceWebGPU } from "./backends/webgpu";
import { inferenceWASM } from "./backends/wasm";
import { inferenceJS } from "./backends/js";

const HEIGHT = 224;
const WIDTH = 224;

const chart = echarts.init(document.getElementById("bench")!);
let option = {
  grid: {
    show: false,
  },
  xAxis: {
    name: "Inference time",
    nameTextStyle: {
      fontSize: 14,
      fontWeight: "bolder",
    },
    nameLocation: "middle",
    type: "value",
    splitLine: {
      show: false,
    },
    axisLine: {
      show: true,
      symbol: ["none", "arrow"],
    },
    axisTick: {
      show: false,
    },
    axisLabel: {
      show: false,
    },
  },
  yAxis: {
    type: "category",
    data: ["JS", "WASM", "WebGPU"],
    axisTick: {
      show: false,
    },
  },
  series: [
    {
      data: [0, 0, 0],
      type: "bar",
      itemStyle: {
        normal: {
          color: "orange",
        },
      },
      label: {
        show: true,
        position: "right",
        formatter: "{c} ms",
      },
    },
  ],
};
chart.setOption(option);

// Main
const btn = document.querySelector("#btn");
btn!.addEventListener("click", async () => {
  // Get file content by filesystem API
  let [fileHandle] = await (window as any).showOpenFilePicker();
  const file = await fileHandle.getFile();
  const content = await file.arrayBuffer();

  // Transform file content to tensor and do preprocessing
  const buffer = Buffer.from(content);
  // Do resize here so we can display resized image later
  const image = (await loadImage(buffer)).resize(HEIGHT, WIDTH);

  const canvas = document.querySelector("canvas")!;
  displayImage(canvas, image);

  let result = await inferenceJS(image, HEIGHT, WIDTH);
  option.series[0].data[0] = result.elapse;
  chart.setOption(option);

  const clazz = imagenetClasses[result.index][1];
  const targetH = document.querySelector("h2")!;
  targetH.innerText = `"${clazz}"`;

  result = await inferenceWASM(image, HEIGHT, WIDTH);
  option.series[0].data[1] = result.elapse;
  chart.setOption(option);

  result = await inferenceWebGPU(image, HEIGHT, WIDTH);
  option.series[0].data[2] = result.elapse;
  chart.setOption(option);
});
