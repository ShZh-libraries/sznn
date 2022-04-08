import { Tensor } from "../tensor";

// Do nothing in inference phase
export function handleDropout(input: Tensor): Tensor[] {
  return [input.copy(), new Tensor()];
}
