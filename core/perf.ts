export interface LayerStat {
    op: string,
    name: string,
    time: number,
}

export type ModelStat = LayerStat[];