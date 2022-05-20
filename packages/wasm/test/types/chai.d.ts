// Declartion merging
declare namespace Chai {
  interface Deep {
    closeTo: (expected: number[], delta: number, msg?: string) => void;
  }
}
