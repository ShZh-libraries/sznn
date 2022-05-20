export function arrayToVec4(arr: number[]) {
  let vec4 = [0, 0, 0, 0];
  for (let i = 0; i < arr.length; i++) {
    vec4[3 - i] = arr[arr.length - 1 - i];
  }

  return vec4;
}
