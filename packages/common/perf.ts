export interface LayerStat {
  op: string;
  name: string;
  time: number;
}

export type ModelStat = LayerStat[];

export function calcOpMeanTime(stats: ModelStat) {
  let opTimesDict = new Map<string, number[]>();

  for (const stat of stats) {
    if (opTimesDict.has(stat.op)) {
      let times = opTimesDict.get(stat.op)!;
      times.push(stat.time);
    } else {
      opTimesDict.set(stat.op, [stat.time]);
    }
  }

  let opStatDict = new Map<string, number>();
  for (const [op, times] of opTimesDict.entries()) {
    let avgTime = times.reduceRight((x, y) => x + y) / times.length;
    opStatDict.set(op, avgTime);
  }

  return opStatDict;
}

export function caclAllTime(stats: ModelStat): number {
  let total = 0;
  for (const stat of stats) {
    total += stat.time;
  }

  return total;
}
