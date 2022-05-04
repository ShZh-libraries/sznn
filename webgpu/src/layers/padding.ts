import padding from "./wgsl/padding.wgsl";
import { computePass, GPUDataEnum, Program, Resource, ResourceType as RType } from "../gpu";
import { Tensor, TensorBuilder } from "../tensor";
import { PaddingAttr } from "../../../core/attr/padding";

export async function handlePadding(input: Tensor, attr: PaddingAttr, device: GPUDevice): Promise<Tensor> {
    let out_shape = [
        input.shape[0], input.shape[1], 
        input.shape[2] + attr.pads[0] + attr.pads[2],
        input.shape[3] + attr.pads[1] + attr.pads[3]
    ];
    let output = TensorBuilder.withShape(out_shape);

    let resources: Resource[] = [
        {
            rtype: RType.InputTensor,
            data: input,
        }, {
            rtype: RType.MetaUInt32Array,
            data: input.shape,
        }, {
            rtype: RType.OutputTensor,
            data: output,
        }, {
            rtype: RType.MetaUInt32Array,
            data: output.shape,
        }, {
            rtype: RType.MetaUInt32Array,
            data: [attr.pads[0], attr.pads[1], attr.pads[2], attr.pads[3]],
        }
    ];
    const program: Program = {
        code: padding,
        entry: "main"
    };
    const result = await computePass(resources, [Math.ceil(output.shape[3] / 16), Math.ceil(output.shape[2] / 16), output.shape[1]], program, device, GPUDataEnum.Float32Array);

    output.data = result;

    return output;
}

