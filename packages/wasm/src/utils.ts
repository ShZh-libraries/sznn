import { Tensor, TensorList } from "./rs/pkg";

export function tensorList(inputs: Tensor[]): TensorList {
  let list = new TensorList();
  for (let input of inputs) {
    list.append(input);
  }

  return list;
}
